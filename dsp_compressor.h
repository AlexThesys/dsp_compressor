#pragma once

/*
Compressor/Limiter.
Features:
	Feed-Forward Design
	Half-wave peak detector
	Threshold (dB) & Ratio controls
	Attack & Release times (ms)
	'Analog/Digital' modes
	Look-ahead buffer. Max look-ahead time (ms) == output buffer size
	Soft Knee control
	Gain reduction is apllied independenlty to each channel
*/

namespace dsp {
namespace dynamics {
	inline void		gain_to_db(__m128& in);
	inline void		db_to_gain(__m128& in);
	inline __m128 	log10f_sse2_unsafe(__m128 val);
	inline __m128 	pow10f_sse2_unsafe(__m128 val);

	enum class compressor_mode {
			digital,
			analog,
	};
	
	struct params {
			fp32		threshold;					// dB	0.0f...-20.0f
			fp32		attack;						// ms	0.05f...200.0f.
			u32			release;					// ms	20...2000
			fp32		ratio;						//		1.0f...20.0f	-- bigger value == more compression
			fp32		out_gain;					// db	0.0f...20.0f
			fp32		knee_width;					//		0.0f...20.0f
			u32			look_ahead_tm;				// ms	4...16
			compressor_mode mode;					//		affects attack/release times
	};

	inline u32 find_next_pow2(u32 v) {
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return	v;
	}

	template <u32 _channels, BOOL _limiter, BOOL _has_lookahead>
	class compressor {
		__m128		_output_gain;
		__m128		_cs;
		fp32		_threshold_db;
		fp32		_knee_width_div2;
		static constexpr u32 num_ch = (_channels + 3) / 4; // 2 for multi, 1 for surround and less

		struct {
			__m128	attack_tm;
			__m128	release_tm;
			__m128	envelope[num_ch];	
		}			_detector;

		struct {
			vector<__m128[num_ch]> buffer;
			u32		read_idx;
			u32		write_idx;
			u32		buf_mask;

			void zero_out_buffer() {
				for (auto &s : buffer)
					mem_zero(s, sizeof(s));
			}
		}			_look_ahead;

	public:
		void		init		(const params& p, u32 out_samples);	// out_samples -- output buffer size

		void		resize_look_ahead(u32 out_samples) {
			if (_has_lookahead) {
				_look_ahead.buffer.resize(find_next_pow2(out_samples));
				_look_ahead.buf_mask = _look_ahead.buffer.size() - 1;
				_look_ahead.zero_out_buffer();
			}
			mem_zero			(_detector.envelope, sizeof(_detector.envelope));
		}
		void		reset		() {
			if (_has_lookahead)
				_look_ahead.zero_out_buffer();
			mem_zero			(_detector.envelope, sizeof(_detector.envelope));
		}

		template <BOOL _interleaved>
		void		process		(audio_buffer<_channels> &buffer, u32 out_channels, audio_buffer<_channels> *sidechain_buf=nullptr);

	private:
		inline __m128 lerp_cs_simd(const __m128 &x0, __m128 x1, __m128 y1, const __m128 &xp, __m128 &mask);

		__m128		detect			(__m128 &in, __m128 &out, u32 ch, const __m128 &diff_0, const __m128 &diff_1);
	};

	template <u32 _channels, BOOL _limiter, BOOL _has_lookahead>
	void compressor<_channels, _limiter, _has_lookahead>::init(const params &p, u32 out_samples)	// out_samples -- output buffer size
	{
		_threshold_db				= p.threshold;
		_output_gain				= _mm_set1_ps(powf(10.0f, p.out_gain / 20.0f));
		_cs							= _mm_set1_ps(_limiter ? 1.0f : (1.0f - 1.0f / _max(1.0f, p.ratio)));
		_knee_width_div2			= _min(_abs(_threshold_db), p.knee_width) * 0.5f;
	
		fp32 tc;
		switch (p.mode) {
		case compressor_mode::digital: tc = log10f(0.01f);
			break;
		case compressor_mode::analog: tc = log10f(0.368f);
			break;
		default: tc					= -1.0f;
			break;
		}
		static constexpr fp32 euler_number = 2.71828f;
		// there is exp(), but it only works with doubles
		_detector.attack_tm			= _mm_set1_ps(powf(euler_number, tc / (_max(1.0f, p.attack * 0.001f * sdef_processing_sample_rate))));
		_detector.release_tm		= _mm_set1_ps(powf(euler_number, tc / ((fp32)p.release * 0.001f * sdef_processing_sample_rate)));
		mem_zero					(_detector.envelope, sizeof(_detector.envelope));
		if (_has_lookahead) {
			_look_ahead.buffer.resize(find_next_pow2(out_samples));
			_look_ahead.buf_mask	= _look_ahead.buffer.size() - 1;
			_look_ahead.zero_out_buffer();
			_look_ahead.read_idx	= 0;
			_look_ahead.write_idx	= (u32)((fp32)p.look_ahead_tm * 0.001f * sdef_processing_sample_rate) & _look_ahead.buf_mask;
		}
	}

	template <u32 _channels, BOOL _limiter, BOOL _has_lookahead>
	template <BOOL _interleaved>
	void compressor<_channels, _limiter, _has_lookahead>::process(audio_buffer<_channels> &buffer, u32 out_channels, audio_buffer<_channels> *sidechain_buf) {
		const u32 num_halves		= _interleaved ? 1 : ((out_channels + 3) / 4);
		const u32 incr				= _interleaved ? 1 : 2;
		verify						(num_halves <= num_ch);
		if (_interleaved && _has_lookahead)
			verify					(buffer.vec_size() <= _look_ahead.buffer.size());
		audio_buffer<_channels> &buf = (sidechain_buf == nullptr) ? buffer : *sidechain_buf;
		verify						(buf.size() == buffer.size());
		// maybe set once in init() and keep as member vars
		const __m128 x0				= _mm_set1_ps(_threshold_db - _knee_width_div2);
		const __m128 x1				= _mm_set1_ps(_min(0.0f, _threshold_db + _knee_width_div2));
		const __m128 threshold_db	= _mm_set1_ps(_threshold_db);
		const __m128 zeroes			= _mm_setzero_ps();
		__m128 detected_db;

		for (u32 hf = 0, sz = buffer.vec_size(); hf < sz; hf += incr) { // hf -- half frame
			for (u32 ch = 0; ch < num_halves; ch++) {
				__m128 input		= buffer.vec_data()[hf + ch].m;
				__m128 detected		= detect(input, detected_db, ch, x0, x1);
				// calculate gain
				if (_has_lookahead)	// compile time branch
					_look_ahead.buffer[_look_ahead.write_idx][ch] = buffer.vec_data()[hf + ch].m;
				__m128 cs			= _cs;
				if (!_limiter && _mm_movemask_ps(detected))	// if was detected in at least one channel
					cs				= lerp_cs_simd(x0, x1, _cs, detected_db, detected);
				__m128 y_gain		= _mm_sub_ps(threshold_db, detected_db);
				if (!_limiter)	// compile time branch
					y_gain			= _mm_mul_ps(y_gain, cs);
				y_gain				= _mm_min_ps(y_gain, zeroes);
				db_to_gain			(y_gain);
				// apply gain
				y_gain				= _mm_mul_ps(y_gain, _output_gain);
				y_gain				= _mm_mul_ps(y_gain, _has_lookahead ? _look_ahead.buffer[_look_ahead.read_idx][ch] : buffer.vec_data()[hf+ch].m);
				buffer.vec_data()[hf + ch].m = y_gain;
			}
			if (_has_lookahead) {
				_look_ahead.read_idx = ++_look_ahead.read_idx & _look_ahead.buf_mask;
				_look_ahead.write_idx = ++_look_ahead.write_idx & _look_ahead.buf_mask;
			}
		}
	}

	template <u32 _channels, BOOL _limiter, BOOL _has_lookahead>
	__m128 compressor<_channels, _limiter, _has_lookahead>::detect(__m128 &in, __m128 &out, u32 ch, const __m128 &diff_0, const __m128 &diff_1) {
		const __m128i all_bits_set	= _mm_set1_epi32(0xFFFFFFFF);
		const __m128i nsb			= _mm_set1_epi32(0x7FFFFFFF);
		const __m128 &not_sign_bit	= *(__m128*)&nsb;
		__m128 detected				= _mm_setzero_ps();

		in							= _mm_and_ps(in, not_sign_bit);	// half-wave detection
		_detector.envelope[ch]		= _mm_sub_ps(_detector.envelope[ch], in);
		__m128i mask				= _mm_srai_epi32(*(__m128i*) & _detector.envelope[ch], 31);
		const __m128 att			= _mm_and_ps(_detector.attack_tm, *(__m128*) & mask);
		mask						= _mm_xor_si128(mask, all_bits_set); // invert mask
		const __m128 rel			= _mm_and_ps(_detector.release_tm, *(__m128*) & mask);
		const __m128 time			= _mm_add_ps(att, rel);
		_detector.envelope[ch]		= _mm_mul_ps(_detector.envelope[ch], time);
		_detector.envelope[ch]		= _mm_add_ps(_detector.envelope[ch], in);

		out							= _detector.envelope[ch];
		gain_to_db					(out);

		if (_knee_width_div2 > 0) {
			const __m128 d_0		= _mm_cmplt_ps(diff_0, out);
			const __m128 d_1		= _mm_cmplt_ps(out, diff_1);
			detected				= _mm_and_ps(d_0, d_1);
		}
		return						detected;
	}

	template <u32 _channels, BOOL _limiter, BOOL _has_lookahead>
	inline __m128 compressor<_channels, _limiter, _has_lookahead>::lerp_cs_simd(const __m128 &x0, __m128 x1, __m128 y1, const __m128 &xp, __m128 &mask) {
		__m128 temp					= _mm_sub_ps(xp, x0);
		x1							= _mm_sub_ps(x1, x0);
		x1							= _mm_div_ps(temp, x1);
		temp						= _mm_mul_ps(y1, x1);
		temp						= _mm_and_ps(temp, mask);
		const __m128i all_bits_set	= _mm_set1_epi32(0xFFFFFFFF);
		mask						= _mm_xor_ps(mask, *(__m128*)&all_bits_set); // invert mask
		y1							= _mm_and_ps(y1, mask);
		return						_mm_add_ps(y1, temp);
	}

	inline void	gain_to_db			(__m128 &in) {
		// another option would be to save the mask, fill zeros with ones, and later invert the mask and add -96.0f (noise floor) after the conversion
		// __m128 eps					= _mm_set1_ps(0.000000001f);
		// in							= _mm_max_ps(in, eps);
		// alignas(16) float buf[4];
		// _mm_store_ps				(buf, in);
		// // compiler is going to use vectorized log10f() 
		// buf[0]						= log10f(buf[0]);
		// buf[1]						= log10f(buf[1]);
		// buf[2]						= log10f(buf[2]);
		// buf[3]						= log10f(buf[3]);
		// in							= _mm_load_ps(buf);
		in 							= log10f_sse2_unsafe(in);
		const __m128 factor			= _mm_set1_ps(20.0f);
		in							= _mm_mul_ps(in, factor);
	}

	inline void	db_to_gain			(__m128 &in) {
		const __m128 divisor_rec	= _mm_set1_ps(1.0f / 20.0f);
		in							= _mm_mul_ps(in, divisor_rec);
		// alignas(16) float buf[4];
		// _mm_store_ps				(buf, in);
		// // compiler is going to use vectorized powf()
		// buf[0]						= powf(10.0f, buf[0]);
		// buf[1]						= powf(10.0f, buf[1]);
		// buf[2]						= powf(10.0f, buf[2]);
		// buf[3]						= powf(10.0f, buf[3]);
		// in							= _mm_load_ps(buf);
		in 							= pow10f_sse2_unsafe(in);
	}
	
	union m128 {
    __m128 	f;
    __m128d d;
    __m128i i;
	};

	inline __m128 log10f_sse2_unsafe(__m128 val) {
		const __m128 eps 			= _mm_set1_ps(0.000000001f);
		const __m128i mantissa_mask = _mm_set1_epi32(0x007fffff);
		const __m128i sqrt_2_addition = _mm_set1_epi32(0x004afb10);
		const __m128i f_one 		= _mm_set1_epi32(0x3f800000);
		const __m128i sqrt_2_mask 	= _mm_set1_epi32(0x00800000);
		const __m128i exp_bias 		= _mm_set1_epi32(0x0000007f);
		const __m128i f_inv_ln10 	= _mm_set1_epi32(0x3ede5bd9);
		const __m128i f_b0 			= _mm_set1_epi32(0x3e943d93);
		const __m128i f_b1 			= _mm_set1_epi32(0x3e319274);
		const __m128i f_b2 			= _mm_set1_epi32(0x3e096bb1);
		const __m128i f_lg2 		= _mm_set1_epi32(0x3e9a209b);
	
		m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5; 
		xmm0.f 						= _mm_max_ps(val, eps);  // input
		xmm1.i 						= _mm_and_si128(xmm0.i, mantissa_mask);
		xmm2.i 						= _mm_add_epi32(sqrt_2_addition, xmm1.i);
		xmm2.i 						= _mm_and_si128(xmm2.i, sqrt_2_mask);
		xmm3.i 						= _mm_xor_si128(f_one, xmm2.i);
		xmm2.i 						= _mm_srai_epi32(xmm2.i, 0x17);
		xmm1.i 						= _mm_or_si128(xmm1.i, xmm3.i);
		xmm0.i 						= _mm_srai_epi32(xmm0.i, 0x17);
		xmm4.f 						= _mm_add_ps(xmm1.f, *(__m128*)&f_one);
		xmm1.f 						= _mm_sub_ps(xmm1.f, *(__m128*)&f_one);
		xmm0.i 						= _mm_sub_epi32(xmm0.i, exp_bias);
		xmm0.i 						= _mm_add_epi32(xmm0.i, xmm2.i);
		xmm0.f 						= _mm_cvtepi32_ps(xmm0.i);
		xmm2.f 						= _mm_mul_ps(*(__m128*)&f_inv_ln10, xmm1.f);
		xmm1.f 						= _mm_div_ps(xmm1.f, xmm4.f);
		xmm4.f 						= xmm1.f;
		xmm1.f 						= _mm_mul_ps(xmm1.f, xmm1.f);
		xmm3.f 						= xmm1.f;
		xmm1.f 						= _mm_mul_ps(xmm1.f, xmm1.f);
		xmm5.f 						= _mm_mul_ps(xmm1.f, *(__m128*)&f_b2);
		xmm1.f 						= _mm_mul_ps(xmm1.f, *(__m128*)&f_b1);
		xmm5.f 						= _mm_add_ps(xmm5.f, *(__m128*)&f_b0);
		xmm5.f 						= _mm_mul_ps(xmm5.f, xmm3.f);
		xmm0.f 						= _mm_mul_ps(xmm0.f, *(__m128*) & f_lg2);
		xmm5.f 						= _mm_add_ps(xmm5.f, xmm1.f);
		xmm5.f 						= _mm_sub_ps(xmm5.f, xmm2.f);
		xmm5.f 						= _mm_mul_ps(xmm5.f, xmm4.f);
		xmm5.f 						= _mm_add_ps(xmm5.f, xmm2.f);
		xmm0.f 						= _mm_add_ps(xmm0.f, xmm5.f);
		return						xmm0.f;
	}
	
	inline __m128 pow10f_sse2_unsafe(__m128 val) {
		const __m128i d_k_0     	= _mm_set_epi32(0x400010FB, 0x6046F1C3, 0x400010FB, 0x6046F1C3);
		const __m128i d_k_3_5   	= _mm_set_epi32(0x400A934F, 0x0979A2A4, 0x400A934F, 0x0979A2A4);
			
		const __m128i d_k_110   	= _mm_set_epi32(0x43380000, 0x00000000, 0x43380000, 0x00000000);
		const __m128i d_k_130   	= _mm_set_epi32(0x00000000, 0x000003f2, 0x00000000, 0x000003f2);
		const __m128i d_k_140   	= _mm_set_epi32(0x401b6eb4, 0x213bde9f, 0x401b6eb4, 0x213bde9f);
		const __m128i d_k_150   	= _mm_set_epi32(0x4010bf17, 0x348d1e8d, 0x4010bf17, 0x348d1e8d);
		const __m128i d_k_160   	= _mm_set_epi32(0x402a6a2a, 0x50304bbe, 0x402a6a2a, 0x50304bbe);
		const __m128i d_k_170   	= _mm_set_epi32(0x4030a032, 0xa22d3a94, 0x4030a032, 0xa22d3a94);
		const __m128i d_k_180   	= _mm_set_epi32(0xc002807e, 0xbb3fa39c, 0xc002807e, 0xbb3fa39c);
		const __m128i d_k_190   	= _mm_set_epi32(0x403dad2e, 0x293ab434, 0x403dad2e, 0x293ab434);
		const __m128i d_k_1A0   	= _mm_set_epi32(0x00041d33, 0x371366f6, 0x00041d33, 0x371366f6);
	
		m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
		
		xmm2.d  					= _mm_cvtps_pd(val);
		xmm6.i  					= _mm_srli_si128(*(__m128i*)&val, 0x08); // logical shift right by 8 bytes
		xmm6.d  					= _mm_cvtps_pd(xmm6.f);
		xmm3.d  					= _mm_mul_pd(*(__m128d*)&d_k_3_5, xmm2.d);
		xmm5.d  					= _mm_mul_pd(*(__m128d*)&d_k_3_5, xmm6.d);
		xmm0.d  					= _mm_add_pd(*(__m128d*)&d_k_110, xmm3.d);
		xmm4.d  					= _mm_sub_pd(*(__m128d*)&d_k_110, xmm0.d);
		xmm3.d  					= _mm_add_pd(xmm3.d, xmm4.d);
		xmm0.i  					= _mm_add_epi32(xmm0.i, d_k_130);
		xmm1.d  					= _mm_add_pd(*(__m128d*)&d_k_110, xmm5.d);
		xmm6.d  					= _mm_sub_pd(*(__m128d*)&d_k_110, xmm1.d);
		xmm5.d  					= _mm_add_pd(xmm5.d, xmm6.d);
		xmm7.d  					= _mm_add_pd(*(__m128d*)&d_k_140, xmm3.d);
		xmm1.i  					= _mm_add_epi32(xmm1.i, d_k_130);
		xmm6.d  					= xmm5.d;
		xmm7.d  					= _mm_mul_pd(xmm7.d, xmm3.d);
		xmm5.d  					= _mm_add_pd(xmm5.d, *(__m128d*)&d_k_150);
		xmm4.d  					= _mm_add_pd(xmm3.d, *(__m128d*)&d_k_150);
		xmm7.d  					= _mm_add_pd(xmm7.d, *(__m128d*)&d_k_160);
		xmm5.d  					= _mm_mul_pd(xmm5.d, xmm6.d);
		xmm4.d  					= _mm_mul_pd(xmm4.d, xmm3.d);
		xmm2.d  					= _mm_add_pd(xmm6.d, *(__m128d*)&d_k_140);
		xmm0.i  					= _mm_slli_epi64(xmm0.i, 0x34);
		xmm5.d  					= _mm_add_pd(xmm5.d, *(__m128d*)&d_k_170);
		xmm4.d  					= _mm_add_pd(xmm4.d, *(__m128d*)&d_k_170);
		xmm2.d  					= _mm_mul_pd(xmm2.d, xmm6.d);
		xmm7.d  					= _mm_mul_pd(xmm7.d, xmm4.d);
		xmm2.d  					= _mm_add_pd(xmm2.d, *(__m128d*)&d_k_160);
		xmm4.d  					= xmm3.d;
		xmm5.d  					= _mm_mul_pd(xmm5.d, xmm2.d);
		xmm3.d  					= _mm_add_pd(xmm3.d, *(__m128d*)&d_k_180);
		xmm1.i  					= _mm_slli_epi64(xmm1.i, 0x34);
		xmm3.d  					= _mm_mul_pd(xmm3.d, xmm4.d);
		xmm2.d  					= _mm_add_pd(xmm6.d, *(__m128d*)&d_k_180);
		xmm3.d  					= _mm_add_pd(xmm3.d, *(__m128d*)&d_k_190);
		xmm2.d  					= _mm_mul_pd(xmm2.d, xmm6.d);
		xmm0.i  					= _mm_or_si128(xmm0.i, d_k_1A0);
		xmm7.d  					= _mm_mul_pd(xmm7.d, xmm3.d);
		xmm2.d  					= _mm_add_pd(xmm2.d, *(__m128d*)&d_k_190);
		xmm0.d  					= _mm_mul_pd(xmm0.d, xmm7.d);
		xmm6.i  					= _mm_or_si128(d_k_1A0, xmm1.i);
		xmm5.d  					= _mm_mul_pd(xmm5.d, xmm2.d);
		xmm0.f  					= _mm_cvtpd_ps(xmm0.d);
		xmm5.d  					= _mm_mul_pd(xmm5.d, xmm6.d);
		xmm5.f  					= _mm_cvtpd_ps(xmm5.d);
		xmm0.f  					= _mm_shuffle_ps(xmm0.f, xmm5.f, _MM_SHUFFLE(1, 0, 1, 0));
		return    					xmm0.f;
	}

}
}

