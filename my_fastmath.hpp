#ifndef MY_FASTMATH_HPP
#define MY_FASTMATH_HPP
/**
   関数リスト
   double my_fastmath::expd_fast (double);
   double my_fastmath::logd_fast (double);

   void   my_fastmath::expd_fast_v(double *py, const double *px, size_t n);
   void   my_fastmath::logd_fast_v(double *py, const double *px, size_t n);
*/

#include <emmintrin.h>
#include <limits>
#include <math.h>
#include <stdint.h>

#ifndef MIE_ALIGN
#ifdef _MSC_VER
#define MIE_ALIGN(x) __declspec(align(x))
#else
#define MIE_ALIGN(x) __attribute__((aligned(x)))
#endif
#endif

#define MY_PI 3.14159265358979323846
#define EXPD_TABLE_SIZE 11
#define LOGD_TABLE_SIZE 11

namespace my_fastmath {
namespace local {
union di {
    double d;
    uint64_t i;
};
inline unsigned int mask(int x) { return (1u << x) - 1; }
inline uint64_t mask64(int x) { return (1ull << x) - 1; }

struct ExpdVar {
    enum { sbit = EXPD_TABLE_SIZE, s = 1ul << sbit, adj = (1ul << (sbit + 10)) - (1ul << sbit) };
    // A = 1, B = 1, C = 1/2, D = 1/6
    double C1; // A
    double C2; // D
    double C3; // C/D
    double tbl[s];
    double a;
    double ra;
    ExpdVar() : a(s / ::log(2.0)), ra(1 / a), C1(1.0), C2(0.16666666685227835064), C3(3.0000000027955394) {
        for (int i = 0; i < s; i++) {
            di dx;
            dx.d = ::pow(2.0, (double)i / s);
            dx.i &= mask64(52);
            tbl[i] = dx.d;
        }
    }
};

struct LogdVar {
    enum { sbit = LOGD_TABLE_SIZE, s = 1ul << sbit };
    double tbl[s];
    LogdVar() {
        for (int i = 0; i < s; i++) {
            tbl[i] = ::log((double)(i + s) / s);
        }
    }
};

// to define static variables in fmath.hpp
struct C {
    static const ExpdVar expdVar;
    static const LogdVar logdVar;
};
MIE_ALIGN(32) const ExpdVar C::expdVar;
MIE_ALIGN(32) const LogdVar C::logdVar;

} // my_fastmath::local

// === 倍精度 指数関数 (スカラ) ============================================
inline double expd_fast(double x) {
    if (x <= -708.39641853226408) {
        return 0.0;
    }
    if (x >= 709.78271289338397) {
        return std::numeric_limits<double>::infinity();
    }
    using namespace local;
    const ExpdVar &c = C::expdVar;
    const double b   = (double)(3ull << 51); // 2^52 + 2^51

    di dx, iax;
    dx.d       = x * c.a + b;
    iax.d      = c.tbl[dx.i & mask(c.sbit)];
    double t   = (dx.d - b) * c.ra - x;
    uint64_t u = ((dx.i + c.adj) >> c.sbit) << 52;
    double y   = (c.C3 - t) * (t * t) * c.C2 - t + c.C1;

    dx.i = u | iax.i;
    return y * dx.d;
}

// === 倍精度 指数関数 (ベクトル) ==========================================
inline void expd_fast_v(double *py, const double *px, size_t n) {
    using namespace local;
    const ExpdVar &c  = C::expdVar;
    const __m128d mb  = _mm_set1_pd(double(3ull << 51));
    const __m128d mC1 = _mm_set1_pd(c.C1);
    const __m128d mC2 = _mm_set1_pd(c.C2);
    const __m128d mC3 = _mm_set1_pd(c.C3);
    const __m128d ma  = _mm_set1_pd(c.a);
    const __m128d mra = _mm_set1_pd(c.ra);
#if defined(__x86_64__) || defined(_WIN64)
    const __m128i madj = _mm_set1_epi64x(c.adj);
#else
    const __m128i madj = _mm_set_epi32(0, c.adj, 0, c.adj);
#endif
    const __m128d expMax    = _mm_set1_pd(709.78272569338397);
    const __m128d expMin    = _mm_set1_pd(-708.39641853226408);
    const __m128i mask_sbit = _mm_set1_epi64x(mask64(c.sbit));

    size_t r = n & 1;
    n &= ~1;

    for (size_t i = 0; i < n; i += 2) {
        __m128d x = _mm_load_pd(px + i);
        x         = _mm_min_pd(x, expMax);
        x         = _mm_max_pd(x, expMin);

        __m128d d = _mm_mul_pd(x, ma);
        d         = _mm_add_pd(d, mb);

        __m128i iax = _mm_castpd_si128(d);
        iax         = _mm_and_si128(iax, mask_sbit);

        int addr0      = _mm_cvtsi128_si32(iax);
        int addr1      = _mm_cvtsi128_si32(_mm_srli_si128(iax, 8));
        __m128d iax_dL = _mm_load_sd(c.tbl + addr0);
        __m128d iax_d  = _mm_load_sd(c.tbl + addr1);
        iax_d          = _mm_unpacklo_pd(iax_dL, iax_d);
        iax            = _mm_castpd_si128(iax_d);

        __m128d t = _mm_sub_pd(_mm_mul_pd(_mm_sub_pd(d, mb), mra), x);
        __m128i u = _mm_castpd_si128(d);
        u         = _mm_add_epi64(u, madj);
        u         = _mm_srli_epi64(u, c.sbit);
        u         = _mm_slli_epi64(u, 52);
        u         = _mm_or_si128(u, iax);
        __m128d y = _mm_mul_pd(_mm_sub_pd(mC3, t), _mm_mul_pd(t, t));
        y         = _mm_mul_pd(y, mC2);
        y         = _mm_add_pd(_mm_sub_pd(y, t), mC1);
        _mm_store_pd(py + i, _mm_mul_pd(y, _mm_castsi128_pd(u)));
    }

    if (r == 1) {
        py[n] = expd_fast(px[n]);
    }
}

// === 倍精度 対数関数 (スカラ) ============================================
inline double logd_fast(double x) {
    if (x <= 0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (x > 1.7976931348623157e+308) {
        return std::numeric_limits<double>::infinity();
    }
    using namespace local;
    const LogdVar &c = C::logdVar;
    di w, kasuu_s;
    double y, h, z;
    w.d       = x;
    kasuu_s.i = w.i & mask64(52);
    int k     = (int)(kasuu_s.i >> (52 - c.sbit)) + c.s;
    kasuu_s.i = kasuu_s.i ^ ((0x3FFull + c.sbit) << 52); // kasuu*2^(sbit)
    int l     = k - c.s;

    h = kasuu_s.d;
    h = (h - k) / (h + k);
    z = h * h;
    y = ((2.0 / 3) * z + 2.0) * h;
    return ((int)(w.i >> 52) - 1023) * ::log(2) + c.tbl[l] + y;
}

// === 倍精度 対数関数 (ベクトル) ==========================================
inline void logd_fast_v(double *py, const double *px, size_t n) {
    using namespace local;
    const LogdVar &c      = C::logdVar;
    const __m128d dblMax  = _mm_set1_pd(1.7976931348623157e+308); // 正の最大の正規化数
    const __m128d dblMin  = _mm_set1_pd(2.2250738585072014e-308); // 正の最小の正規化数
    const __m128d f_mask  = _mm_set1_pd(2.2250738585072009e-308); // 仮数部を取り出すマスク
    const __m128d md_S    = _mm_set1_pd((double)c.s);
    const __m128i mi_S    = _mm_set_epi32(0, 0, c.s, c.s);
    const __m128d md_C3   = _mm_set1_pd(2.0 / 3);
    const __m128d md_C1   = _mm_set1_pd(2.0);
    const __m128d md_LOG2 = _mm_set1_pd(::log(2));
    const __m128i mi_1023 = _mm_set_epi32(0, 0, 1023, 1023);

    size_t r = n & 1; // 2で割った時の剰余
    n &= ~1;          // 2で割った時の商

    for (size_t i = 0; i < n; i += 2) {
        // メモリからロード
        __m128d x = _mm_load_pd(px + i);
        x         = _mm_min_pd(x, dblMax);
        x         = _mm_max_pd(x, dblMin);

        // 仮数部の計算
        __m128d md_H = _mm_and_pd(x, f_mask); // 仮数部を取り出す.
        __m128i mi_K = _mm_castpd_si128(md_H);
        md_H         = _mm_xor_pd(md_H, md_S); // 仮数 * 2^(sbit)

        mi_K            = _mm_srli_epi64(mi_K, (52 - c.sbit));              // R3, R2, R1, R0
        mi_K            = _mm_shuffle_epi32(mi_K, _MM_SHUFFLE(3, 1, 2, 0)); // R3, R1, R2, R0
        mi_K            = _mm_xor_si128(mi_K, mi_S);
        __m128i mi_L    = _mm_sub_epi32(mi_K, mi_S);
        int addr0       = _mm_cvtsi128_si32(mi_L);
        int addr1       = _mm_cvtsi128_si32(_mm_srli_si128(mi_L, 4));
        __m128d tbl_l   = _mm_load_sd(c.tbl + addr0);
        __m128d tbl_tmp = _mm_load_sd(c.tbl + addr1);
        tbl_l           = _mm_unpacklo_pd(tbl_l, tbl_tmp);

        __m128d md_K = _mm_cvtepi32_pd(mi_K);
        __m128d md_Y = _mm_sub_pd(md_H, md_K);
        __m128d md_Z = _mm_add_pd(md_H, md_K);
        md_H         = _mm_div_pd(md_Y, md_Z); // h = (h - k) / (h + k)
        md_Z         = _mm_mul_pd(md_H, md_H); // z = h^2
        md_Y         = _mm_mul_pd(md_C3, md_Z);
        md_Y         = _mm_add_pd(md_Y, md_C1);
        md_Y         = _mm_mul_pd(md_Y, md_H);

        md_Y = _mm_add_pd(md_Y, tbl_l);

        // 指数部の計算
        __m128i mi_N = _mm_castpd_si128(x);

        // R3 = px[1]の指数, R2 = 0, R1 = px[0]の指数, R0 = 0
        mi_N = _mm_srli_epi32(mi_N, 20);
        // R2, R0, R3, R1 (※上記の文字を使用)
        mi_N         = _mm_shuffle_epi32(mi_N, _MM_SHUFFLE(2, 0, 3, 1));
        mi_N         = _mm_sub_epi32(mi_N, mi_1023);
        __m128d md_N = _mm_cvtepi32_pd(mi_N);

        // 全体の計算
        md_N = _mm_mul_pd(md_N, md_LOG2);
        md_Y = _mm_add_pd(md_N, md_Y);

        // メモリへストア
        _mm_store_pd(py + i, md_Y);
    }

    if (r == 1) {
        py[n] = logd_fast(px[n]);
    }
}

} // my_fastmath

#undef MY_PI
#undef EXPD_TABLE_SIZE
#undef LOGD_TABLE_SIZE
#endif // MY_FASTMATH_HPP
