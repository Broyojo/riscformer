/* Division helpers without hardware div (the core implements M multiply only;
 * we link with -mno-div and these preempt libgcc's div-instruction versions). */
#include <stdint.h>

uint32_t __udivmodsi4(uint32_t n, uint32_t d, uint32_t *rem) {
    uint32_t q = 0, r = 0;
    if (d) {
        for (int i = 31; i >= 0; i--) {
            r = (r << 1) | ((n >> i) & 1);
            if (r >= d) { r -= d; q |= 1u << i; }
        }
    } else {
        q = 0xFFFFFFFFu; r = n;
    }
    if (rem) *rem = r;
    return q;
}
uint32_t __udivsi3(uint32_t n, uint32_t d) { return __udivmodsi4(n, d, 0); }
uint32_t __umodsi3(uint32_t n, uint32_t d) { uint32_t r; __udivmodsi4(n, d, &r); return r; }
int32_t __divsi3(int32_t n, int32_t d) {
    int neg = (n < 0) ^ (d < 0);
    uint32_t q = __udivmodsi4(n < 0 ? -(uint32_t)n : n, d < 0 ? -(uint32_t)d : d, 0);
    return neg ? -(int32_t)q : (int32_t)q;
}
int32_t __modsi3(int32_t n, int32_t d) {
    uint32_t r;
    __udivmodsi4(n < 0 ? -(uint32_t)n : n, d < 0 ? -(uint32_t)d : d, &r);
    return n < 0 ? -(int32_t)r : (int32_t)r;
}

uint64_t __udivmoddi4(uint64_t n, uint64_t d, uint64_t *rem) {
    uint64_t q = 0, r = 0;
    if (d) {
        for (int i = 63; i >= 0; i--) {
            r = (r << 1) | ((n >> i) & 1);
            if (r >= d) { r -= d; q |= 1ull << i; }
        }
    } else {
        q = ~0ull; r = n;
    }
    if (rem) *rem = r;
    return q;
}
uint64_t __udivdi3(uint64_t n, uint64_t d) { return __udivmoddi4(n, d, 0); }
uint64_t __umoddi3(uint64_t n, uint64_t d) { uint64_t r; __udivmoddi4(n, d, &r); return r; }
int64_t __divdi3(int64_t n, int64_t d) {
    int neg = (n < 0) ^ (d < 0);
    uint64_t q = __udivmoddi4(n < 0 ? -(uint64_t)n : n, d < 0 ? -(uint64_t)d : d, 0);
    return neg ? -(int64_t)q : (int64_t)q;
}
int64_t __moddi3(int64_t n, int64_t d) {
    uint64_t r;
    __udivmoddi4(n < 0 ? -(uint64_t)n : n, d < 0 ? -(uint64_t)d : d, &r);
    return n < 0 ? -(int64_t)r : (int64_t)r;
}
