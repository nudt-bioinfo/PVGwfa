#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "gwfa.h"
#include "kalloc.h"
#include "ksort.h"
#include <immintrin.h>
#include "global_variants.h"



extern double total_s_extend_time =0, total_p_extend_time=0;
extern double total_s_expand_time =0, total_p_expand_time=0;
extern double total_s_intfinding_time=0, total_p_intfinding_time=0;
extern double total_s_lcp_time=0, total_p_lcp_time=0;





#define kmalloc(km, size) malloc((size))
#define kcalloc(km, count, size) calloc((count), (size))
#define krealloc(km, ptr, size) realloc((ptr), (size))
#define kfree(km, ptr) free((ptr))
/**********************
 * Indexing the graph *
 **********************/

#define arc_key(x) ((x).a)
KRADIX_SORT_INIT(gwf_arc, gwf_arc_t, arc_key, 8)

// index the graph such that we can quickly access the neighbors of a vertex
void gwf_ed_index_arc_core(uint64_t *idx, uint32_t n_vtx, uint32_t n_arc, gwf_arc_t *arc)
{
	uint32_t i, st;
	radix_sort_gwf_arc(arc, arc + n_arc);
	for (st = 0, i = 1; i <= n_arc; ++i) {
		if (i == n_arc || arc[i].a>>32 != arc[st].a>>32) {
			uint32_t v = arc[st].a>>32;
			assert(v < n_vtx);
			idx[v] = (uint64_t)st << 32 | (i - st);
			st = i;
		}
	}
}

void gwf_ed_index(void *km, gwf_graph_t *g)
{
	KMALLOC(km, g->aux, g->n_vtx);
	gwf_ed_index_arc_core(g->aux, g->n_vtx, g->n_arc, g->arc);
}

// free the index
void gwf_cleanup(void *km, gwf_graph_t *g)
{
	kfree(km, g->aux);
	g->aux = 0;
}

/**************************************
 * Graph WaveFront with edit distance *
 **************************************/

#include "khashl.h" // make it compatible with kalloc
#include "kdq.h"
#include "kvec.h"

#define GWF_DIAG_SHIFT 0x40000000

static inline uint64_t gwf_gen_vd(uint32_t v, int32_t d)
{
	return (uint64_t)v<<32 | (GWF_DIAG_SHIFT + d);
}

/*
 * Diagonal interval
 */
typedef struct {
	uint64_t vd0, vd1;
} gwf_intv_t;

typedef kvec_t(gwf_intv_t) gwf_intv_v;

#define intvd_key(x) ((x).vd0)
KRADIX_SORT_INIT(gwf_intv, gwf_intv_t, intvd_key, 8)

static int gwf_intv_is_sorted(int32_t n_a, const gwf_intv_t *a)
{
	int32_t i;
	for (i = 1; i < n_a; ++i)
		if (a[i-1].vd0 > a[i].vd0) break;
	return (i == n_a);
}

void gwf_ed_print_intv(size_t n, gwf_intv_t *a) // for debugging only
{
	size_t i;
	for (i = 0; i < n; ++i)
		printf("Z\t%d\t%d\t%d\n", (int32_t)(a[i].vd0>>32), (int32_t)a[i].vd0 - GWF_DIAG_SHIFT, (int32_t)a[i].vd1 - GWF_DIAG_SHIFT);
}

// merge overlapping intervals; input must be sorted
static size_t gwf_intv_merge_adj(size_t n, gwf_intv_t *a)
{
	size_t i, k;
	uint64_t st, en;
	if (n == 0) return 0;
	st = a[0].vd0, en = a[0].vd1;
	for (i = 1, k = 0; i < n; ++i) {
		if (a[i].vd0 > en) {
			a[k].vd0 = st, a[k++].vd1 = en;
			st = a[i].vd0, en = a[i].vd1;
		} else en = en > a[i].vd1? en : a[i].vd1;
	}
	a[k].vd0 = st, a[k++].vd1 = en;
	return k;
}

// merge two sorted interval lists
static size_t gwf_intv_merge2(gwf_intv_t *a, size_t n_b, const gwf_intv_t *b, size_t n_c, const gwf_intv_t *c)
{
	size_t i = 0, j = 0, k = 0;
	while (i < n_b && j < n_c) {
		if (b[i].vd0 <= c[j].vd0)
			a[k++] = b[i++];
		else a[k++] = c[j++];
	}
	while (i < n_b) a[k++] = b[i++];
	while (j < n_c) a[k++] = c[j++];
	return gwf_intv_merge_adj(k, a);
}

/*
 * Diagonal
 */
typedef struct { // a diagonal
	uint64_t vd; // higher 32 bits: vertex ID; lower 32 bits: diagonal+0x4000000
	int32_t k;
	uint32_t xo; // higher 31 bits: anti diagonal; lower 1 bit: out-of-order or not
	int32_t t;
} gwf_diag_t;

typedef kvec_t(gwf_diag_t) gwf_diag_v;

#define ed_key(x) ((x).vd)
KRADIX_SORT_INIT(gwf_ed, gwf_diag_t, ed_key, 8)

KDQ_INIT(gwf_diag_t)

void gwf_ed_print_diag(size_t n, gwf_diag_t *a) // for debugging only
{
	size_t i;
	for (i = 0; i < n; ++i) {
		int32_t d = (int32_t)a[i].vd - GWF_DIAG_SHIFT;
		printf("Z\t%d\t%d\t%d\t%d\t%d\n", (int32_t)(a[i].vd>>32), d, a[i].k, d + a[i].k, a[i].xo>>1);
	}
}

// push (v,d,k) to the end of the queue
static inline void gwf_diag_push(void *km, gwf_diag_v *a, uint32_t v, int32_t d, int32_t k, uint32_t x, uint32_t ooo, int32_t t)
{
	gwf_diag_t *p;
	kv_pushp(gwf_diag_t, km, *a, &p);
	p->vd = gwf_gen_vd(v, d), p->k = k, p->xo = x<<1|ooo, p->t = t;
}

// determine the wavefront on diagonal (v,d)
static inline int32_t gwf_diag_update(gwf_diag_t *p, uint32_t v, int32_t d, int32_t k, uint32_t x, uint32_t ooo, int32_t t)
{
	uint64_t vd = gwf_gen_vd(v, d);
	if (p->vd == vd) {
		p->xo = p->k > k? p->xo : x<<1|ooo;
		p->t  = p->k > k? p->t : t;
		p->k  = p->k > k? p->k : k;
		return 0;
	}
	return 1;
}

static int gwf_diag_is_sorted(int32_t n_a, const gwf_diag_t *a)
{
	int32_t i;
	for (i = 1; i < n_a; ++i)
		if (a[i-1].vd > a[i].vd) break;
	return (i == n_a);
}

// sort a[]. This uses the gwf_diag_t::ooo field to speed up sorting.
static void gwf_diag_sort(int32_t n_a, gwf_diag_t *a, void *km, gwf_diag_v *ooo)
{
	int32_t i, j, k, n_b, n_c;
	gwf_diag_t *b, *c;

	kv_resize(gwf_diag_t, km, *ooo, n_a);
	for (i = 0, n_c = 0; i < n_a; ++i)
		if (a[i].xo&1) ++n_c;
	n_b = n_a - n_c;
	b = ooo->a, c = b + n_b;
	for (i = j = k = 0; i < n_a; ++i) {
		if (a[i].xo&1) c[k++] = a[i];
		else b[j++] = a[i];
	}
	radix_sort_gwf_ed(c, c + n_c);
	for (k = 0; k < n_c; ++k) c[k].xo &= 0xfffffffeU;

	i = j = k = 0;
	while (i < n_b && j < n_c) {
		if (b[i].vd <= c[j].vd)
			a[k++] = b[i++];
		else a[k++] = c[j++];
	}
	while (i < n_b) a[k++] = b[i++];
	while (j < n_c) a[k++] = c[j++];
}

// remove diagonals not on the wavefront
static int32_t gwf_diag_dedup(int32_t n_a, gwf_diag_t *a, void *km, gwf_diag_v *ooo)
{
	int32_t i, n, st;
	if (!gwf_diag_is_sorted(n_a, a))
		gwf_diag_sort(n_a, a, km, ooo);
	for (i = 1, st = 0, n = 0; i <= n_a; ++i) {
		if (i == n_a || a[i].vd != a[st].vd) {
			int32_t j, max_j = st;
			if (st + 1 < i)
				for (j = st + 1; j < i; ++j) // choose the far end (i.e. the wavefront)
					if (a[max_j].k < a[j].k) max_j = j;
			a[n++] = a[max_j];
			st = i;
		}
	}
	return n;
}

// use forbidden bands to remove diagonals not on the wavefront
static int32_t gwf_mixed_dedup(int32_t n_a, gwf_diag_t *a, int32_t n_b, gwf_intv_t *b)
{
	int32_t i = 0, j = 0, k = 0;
	while (i < n_a && j < n_b) {
		if (a[i].vd >= b[j].vd0 && a[i].vd < b[j].vd1) ++i;
		else if (a[i].vd >= b[j].vd1) ++j;
		else a[k++] = a[i++];
	}
	while (i < n_a) a[k++] = a[i++];
	return k;
}

/*
 * Traceback stack
 */
KHASHL_MAP_INIT(KH_LOCAL, gwf_map64_t, gwf_map64, uint64_t, int32_t, kh_hash_uint64, kh_eq_generic)

typedef struct {
	int32_t v;
	int32_t pre;
} gwf_trace_t;

typedef kvec_t(gwf_trace_t) gwf_trace_v;

static int32_t gwf_trace_push(void *km, gwf_trace_v *a, int32_t v, int32_t pre, gwf_map64_t *h)
{
	uint64_t key = (uint64_t)v << 32 | (uint32_t)pre;
	khint_t k;
	int absent;
	k = gwf_map64_put(h, key, &absent);
	if (absent) {
		gwf_trace_t *p;
		kv_pushp(gwf_trace_t, km, *a, &p);
		p->v = v, p->pre = pre;
		kh_val(h, k) = a->n - 1;
		return a->n - 1;
	}
	return kh_val(h, k);
}

/*
 * Core GWFA routine
 */
KHASHL_INIT(KH_LOCAL, gwf_set64_t, gwf_set64, uint64_t, kh_hash_dummy, kh_eq_generic)

typedef struct {
	void *km;
	gwf_set64_t *ha; // hash table for adjacency
	gwf_map64_t *ht; // hash table for traceback
	gwf_intv_v intv;
	gwf_intv_v tmp, swap;
	gwf_diag_v ooo;
	gwf_trace_v t;
} gwf_edbuf_t;

// remove diagonals not on the wavefront
static int32_t gwf_dedup(gwf_edbuf_t *buf, int32_t n_a, gwf_diag_t *a)
{
	if (buf->intv.n + buf->tmp.n > 0) {
		if (!gwf_intv_is_sorted(buf->tmp.n, buf->tmp.a))
			radix_sort_gwf_intv(buf->tmp.a, buf->tmp.a + buf->tmp.n);
		kv_copy(gwf_intv_t, buf->km, buf->swap, buf->intv);
		kv_resize(gwf_intv_t, buf->km, buf->intv, buf->intv.n + buf->tmp.n);
		buf->intv.n = gwf_intv_merge2(buf->intv.a, buf->swap.n, buf->swap.a, buf->tmp.n, buf->tmp.a);
	}
	n_a = gwf_diag_dedup(n_a, a, buf->km, &buf->ooo);
	if (buf->intv.n > 0)
		n_a = gwf_mixed_dedup(n_a, a, buf->intv.n, buf->intv.a);
	return n_a;
}

// remove diagonals that lag far behind the furthest wavefront
static int32_t gwf_prune(int32_t n_a, gwf_diag_t *a, uint32_t max_lag)
{
	int32_t i, j;
	uint32_t max_x = 0;
	for (i = 0; i < n_a; ++i)
		max_x = max_x > a[i].xo>>1? max_x : a[i].xo>>1;
	if (max_x <= max_lag) return n_a; // no filtering
	for (i = j = 0; i < n_a; ++i)
		if ((a[i].xo>>1) + max_lag >= max_x)
			a[j++] = a[i];
	return j;
}

// reach the wavefront
static inline int32_t gwf_extend1(int32_t d, int32_t k, int32_t vl, const char *ts, int32_t ql, const char *qs)
{
    struct timeval st_s_lcp, en_s_lcp;
//    gettimeofday(&st_s_lcp,NULL);
	int32_t max_k = (ql - d < vl? ql - d : vl) - 1;
	const char *ts_ = ts + 1, *qs_ = qs + d + 1;
#if 0
	// int32_t i = k + d; while (k + 1 < g->len[v] && i + 1 < ql && g->seq[v][k+1] == q[i+1]) ++k, ++i;
	while (k < max_k && *(ts_ + k) == *(qs_ + k))
		++k;
#else
	uint64_t cmp = 0;
	while (k + 7 < max_k) {
		uint64_t x = *(uint64_t*)(ts_ + k); // warning: unaligned memory access
		uint64_t y = *(uint64_t*)(qs_ + k);
		cmp = x ^ y;
		if (cmp == 0) k += 8;
		else break;
	}
	if (cmp)
		k += __builtin_ctzl(cmp) >> 3; // on x86, this is done via the BSR instruction: https://www.felixcloutier.com/x86/bsr
	else if (k + 7 >= max_k)
		while (k < max_k && *(ts_ + k) == *(qs_ + k)) // use this for generic CPUs. It is slightly faster than the unoptimized version
			++k;
#endif
//    gettimeofday(&en_s_lcp,NULL);
//    total_s_lcp_time += (double)(en_s_lcp.tv_sec - st_s_lcp.tv_sec) + (double)(en_s_lcp.tv_usec - st_s_lcp.tv_usec) / 1000000.0;
    return k;
}

static inline int32_t gwf_extend1_vec(int32_t d, int32_t k, int32_t vl, const char *ts, int32_t ql, const char *qs)
{
    struct timeval st_p_lcp, en_p_lcp;
//    gettimeofday(&st_p_lcp,NULL);

    int32_t max_k = (ql - d < vl ? ql - d : vl) - 1;
    const char *ts_ = ts + 1, *qs_ = qs + d + 1;

    int32_t i;
    for (i = 0; i <= max_k - 31; i += 32) {
        __m256i ts_vec = _mm256_loadu_si256((__m256i *)(ts_ + k + i));
        __m256i qs_vec = _mm256_loadu_si256((__m256i *)(qs_ + k + i));
        __m256i cmp = _mm256_cmpeq_epi8(ts_vec, qs_vec);
        uint32_t mask = _mm256_movemask_epi8(cmp);
        if (mask != 0xFFFFFFFF) {
            int32_t index = __builtin_ctz(~mask);
//            gettimeofday(&en_p_lcp,NULL);
//            total_p_lcp_time += (double)(en_p_lcp.tv_sec - st_p_lcp.tv_sec) + (double)(en_p_lcp.tv_usec - st_p_lcp.tv_usec) / 1000000.0;
            return k + i + index;
        }
    }

    for (; i <= max_k - 7; i += 8) {
        __m128i ts_vec = _mm_loadu_si128((__m128i *)(ts_ + k + i));
        __m128i qs_vec = _mm_loadu_si128((__m128i *)(qs_ + k + i));
        __m128i cmp = _mm_cmpeq_epi8(ts_vec, qs_vec);
        uint16_t mask = _mm_movemask_epi8(cmp);
        if (mask != 0xFFFF) {
            int32_t index = __builtin_ctz(~mask);
//            gettimeofday(&en_p_lcp,NULL);
//            total_p_lcp_time += (double)(en_p_lcp.tv_sec - st_p_lcp.tv_sec) + (double)(en_p_lcp.tv_usec - st_p_lcp.tv_usec) / 1000000.0;
            return k + i + index;
        }
    }

    for (; i <= max_k; ++i) {
        if (*(ts_ + k + i) != *(qs_ + k + i)) {
//            gettimeofday(&en_p_lcp,NULL);
//            total_p_lcp_time += (double)(en_p_lcp.tv_sec - st_p_lcp.tv_sec) + (double)(en_p_lcp.tv_usec - st_p_lcp.tv_usec) / 1000000.0;
            return k + i;
        }
    }
//    gettimeofday(&en_p_lcp,NULL);
//    total_p_lcp_time += (double)(en_p_lcp.tv_sec - st_p_lcp.tv_sec) + (double)(en_p_lcp.tv_usec - st_p_lcp.tv_usec) / 1000000.0;
    return max_k;
}

// This is essentially Landau-Vishkin for linear sequences. The function speeds up alignment to long vertices. Not really necessary.
static void gwf_ed_extend_batch(void *km, const gwf_graph_t *g, int32_t ql, const char *q, int32_t n, gwf_diag_t *a, gwf_diag_v *B,
								kdq_t(gwf_diag_t) *A, gwf_intv_v *tmp_intv)
{
    struct timeval st_s_expand, en_s_expand;
//    gettimeofday(&st_s_expand,NULL);

	int32_t j, m;
	int32_t v = a->vd>>32;
	int32_t vl = g->len[v];
	const char *ts = g->seq[v];
	gwf_diag_t *b;

	// wfa_extend
	for (j = 0; j < n; ++j) {
		int32_t k;
		k = gwf_extend1((int32_t)a[j].vd - GWF_DIAG_SHIFT, a[j].k, vl, ts, ql, q);
		a[j].xo += (k - a[j].k) << 2;
		a[j].k = k;
	}

	// wfa_next
	kv_resize(gwf_diag_t, km, *B, B->n + n + 2);
	b = &B->a[B->n];
	b[0].vd = a[0].vd - 1;
	b[0].xo = a[0].xo + 2; // 2 == 1<<1
	b[0].k = a[0].k + 1;
	b[0].t = a[0].t;
	b[1].vd = a[0].vd;
	b[1].xo =  n == 1 || a[0].k > a[1].k? a[0].xo + 4 : a[1].xo + 2;
	b[1].t  =  n == 1 || a[0].k > a[1].k? a[0].t : a[1].t;
	b[1].k  = (n == 1 || a[0].k > a[1].k? a[0].k : a[1].k) + 1;
	for (j = 1; j < n - 1; ++j) {
		uint32_t x = a[j-1].xo + 2;
		int32_t k = a[j-1].k, t = a[j-1].t;
		x = k > a[j].k + 1? x : a[j].xo + 4;
		t = k > a[j].k + 1? t : a[j].t;
		k = k > a[j].k + 1? k : a[j].k + 1;
		x = k > a[j+1].k + 1? x : a[j+1].xo + 2;
		t = k > a[j+1].k + 1? t : a[j+1].t;
		k = k > a[j+1].k + 1? k : a[j+1].k + 1;
		b[j+1].vd = a[j].vd, b[j+1].k = k, b[j+1].xo = x, b[j+1].t = t;
	}
	if (n >= 2) {
		b[n].vd = a[n-1].vd;
		b[n].xo = a[n-2].k > a[n-1].k + 1? a[n-2].xo + 2 : a[n-1].xo + 4;
		b[n].t  = a[n-2].k > a[n-1].k + 1? a[n-2].t : a[n-1].t;
		b[n].k  = a[n-2].k > a[n-1].k + 1? a[n-2].k : a[n-1].k + 1;
	}
	b[n+1].vd = a[n-1].vd + 1;
	b[n+1].xo = a[n-1].xo + 2;
	b[n+1].t  = a[n-1].t;
	b[n+1].k  = a[n-1].k;

	// drop out-of-bound cells
	for (j = 0; j < n; ++j) {
		gwf_diag_t *p = &a[j];
		if (p->k == vl - 1 || (int32_t)p->vd - GWF_DIAG_SHIFT + p->k == ql - 1)
			p->xo |= 1, *kdq_pushp(gwf_diag_t, A) = *p;
	}
	for (j = 0, m = 0; j < n + 2; ++j) {
		gwf_diag_t *p = &b[j];
		int32_t d = (int32_t)p->vd - GWF_DIAG_SHIFT;
		if (d + p->k < ql && p->k < vl) {
			b[m++] = *p;
		} else if (p->k == vl) {
			gwf_intv_t *q;
			kv_pushp(gwf_intv_t, km, *tmp_intv, &q);
			q->vd0 = gwf_gen_vd(v, d), q->vd1 = q->vd0 + 1;
		}
	}
	B->n += m;
//    gettimeofday(&en_s_expand,NULL);
//    total_s_expand_time += (double)(en_s_expand.tv_sec - st_s_expand.tv_sec) + (double)(en_s_expand.tv_usec - st_s_expand.tv_usec) / 1000000.0;

}
static inline void gwf_diag_push_vec(void *km, uint32_t q1_num, gwf_diag_v *a, uint32_t *v, int32_t *d, int32_t *k, uint32_t *xo, int32_t *t, int32_t *mask) {
    if (q1_num == 8) {//eq 8
//        printf("print d\n");
//        for (int i = 0; i < 8; i++) {
//            printf("%d ", d[i] - GWF_DIAG_SHIFT);
//        }
//        printf("end print\n");
        gwf_diag_t *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8;
        kv_pushp_8(gwf_diag_t, km, *a, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8);
        p1->vd = gwf_gen_vd(v[0], d[0] - GWF_DIAG_SHIFT),p1->k = k[0],p1->xo = xo[0],p1->t = t[0];
        p2->vd = gwf_gen_vd(v[1], d[1] - GWF_DIAG_SHIFT),p2->k = k[1],p2->xo = xo[1],p2->t = t[1];
        p3->vd = gwf_gen_vd(v[2], d[2] - GWF_DIAG_SHIFT),p3->k = k[2],p3->xo = xo[2],p3->t = t[2];
        p4->vd = gwf_gen_vd(v[3], d[3] - GWF_DIAG_SHIFT),p4->k = k[3],p4->xo = xo[3],p4->t = t[3];
        p5->vd = gwf_gen_vd(v[4], d[4] - GWF_DIAG_SHIFT),p5->k = k[4],p5->xo = xo[4],p5->t = t[4];
        p6->vd = gwf_gen_vd(v[5], d[5] - GWF_DIAG_SHIFT),p6->k = k[5],p6->xo = xo[5],p6->t = t[5];
        p7->vd = gwf_gen_vd(v[6], d[6] - GWF_DIAG_SHIFT),p7->k = k[6],p7->xo = xo[6],p7->t = t[6];
        p8->vd = gwf_gen_vd(v[7], d[7] - GWF_DIAG_SHIFT),p8->k = k[7],p8->xo = xo[7],p8->t = t[7];
    } else{
//        printf("branch else b\n");
        for (int32_t i = 0; i < q1_num && i < 8; i++) {
            if (mask[i] == 0) continue;
//            printf("%d ", d[i] - GWF_DIAG_SHIFT);
            gwf_diag_t *p1;
            kv_pushp(gwf_diag_t, km, *a, &p1);
            p1->vd = gwf_gen_vd(v[i], d[i] - GWF_DIAG_SHIFT) ,p1->k = k[i],p1->xo = xo[i],p1->t = t[i];
        }
//        printf("branch else e\n");
    }
}
static inline void gwf_ed_extend_batch_vec(void *km, const gwf_graph_t *g, int32_t ql, const char *q, int32_t n, gwf_diag_t *a, gwf_diag_v *B,
                                           kdq_t(gwf_diag_t) *A, gwf_intv_v *tmp_intv)
{
    struct timeval st_p_expand, en_p_expand;
//    gettimeofday(&st_p_expand, NULL);
    int32_t j, m;
    __m256i v_vec = _mm256_set1_epi32(a->vd>>32);
    __m256i vl_vec = _mm256_set1_epi32(g->len[a->vd>>32]);
    int32_t v = a->vd>>32; //vertex id
    int32_t vl = g->len[v];
    const char *ts = g->seq[a->vd>>32];
    gwf_diag_t *b;
    __m256i ak_vec = _mm256_set_epi32(a[7].k,a[6].k,a[5].k,a[4].k,a[3].k,a[2].k,a[1].k,a[0].k);
//    int32_t k0 = gwf_extend1((int32_t)a[0].vd - GWF_DIAG_SHIFT, a[0].k, g->len[a->vd>>32], ts, ql, q);
//    int32_t k1 = gwf_extend1((int32_t)a[1].vd - GWF_DIAG_SHIFT, a[1].k, g->len[a->vd>>32], ts, ql, q);
//    int32_t k2 = gwf_extend1((int32_t)a[2].vd - GWF_DIAG_SHIFT, a[2].k, g->len[a->vd>>32], ts, ql, q);
//    int32_t k3 = gwf_extend1((int32_t)a[3].vd - GWF_DIAG_SHIFT, a[3].k, g->len[a->vd>>32], ts, ql, q);
//    int32_t k4 = gwf_extend1((int32_t)a[4].vd - GWF_DIAG_SHIFT, a[4].k, g->len[a->vd>>32], ts, ql, q);
//    int32_t k5 = gwf_extend1((int32_t)a[5].vd - GWF_DIAG_SHIFT, a[5].k, g->len[a->vd>>32], ts, ql, q);
//    int32_t k6 = gwf_extend1((int32_t)a[6].vd - GWF_DIAG_SHIFT, a[6].k, g->len[a->vd>>32], ts, ql, q);
//    int32_t k7 = gwf_extend1((int32_t)a[7].vd - GWF_DIAG_SHIFT, a[7].k, g->len[a->vd>>32], ts, ql, q);

    int32_t k0 = gwf_extend1_vec((int32_t)a[0].vd - GWF_DIAG_SHIFT, a[0].k, g->len[a->vd>>32], ts, ql, q);
    int32_t k1 = gwf_extend1_vec((int32_t)a[1].vd - GWF_DIAG_SHIFT, a[1].k, g->len[a->vd>>32], ts, ql, q);
    int32_t k2 = gwf_extend1_vec((int32_t)a[2].vd - GWF_DIAG_SHIFT, a[2].k, g->len[a->vd>>32], ts, ql, q);
    int32_t k3 = gwf_extend1_vec((int32_t)a[3].vd - GWF_DIAG_SHIFT, a[3].k, g->len[a->vd>>32], ts, ql, q);
    int32_t k4 = gwf_extend1_vec((int32_t)a[4].vd - GWF_DIAG_SHIFT, a[4].k, g->len[a->vd>>32], ts, ql, q);
    int32_t k5 = gwf_extend1_vec((int32_t)a[5].vd - GWF_DIAG_SHIFT, a[5].k, g->len[a->vd>>32], ts, ql, q);
    int32_t k6 = gwf_extend1_vec((int32_t)a[6].vd - GWF_DIAG_SHIFT, a[6].k, g->len[a->vd>>32], ts, ql, q);
    int32_t k7 = gwf_extend1_vec((int32_t)a[7].vd - GWF_DIAG_SHIFT, a[7].k, g->len[a->vd>>32], ts, ql, q);

    __m256i one_vec = _mm256_set1_epi32(1);
    __m256i shift1_mask = _mm256_set_epi32(0,7,6,5,4,3,2,1);
    __m256i shift2_mask = _mm256_set_epi32(1,0,7,6,5,4,3,2);
    __m256i k_vec =  _mm256_set_epi32(k7,k6,k5,k4,k3,k2,k1,k0);
    __m256i k_j1_vec = _mm256_and_si256(_mm256_permutevar8x32_epi32(k_vec, shift1_mask), _mm256_set_epi32(0,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff));
    __m256i k_j2_vec = _mm256_and_si256(_mm256_permutevar8x32_epi32(k_vec, shift2_mask),_mm256_set_epi32(0,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff));
//    __m256i k_j1_vec = _mm256_set_epi32(0,k7,k6,k5,k4,k3,k2,k1);
//    __m256i k_j2_vec =  _mm256_set_epi32(0,0,k7,k6,k5,k4,k3,k2);
    __m256i kp1_vec =  _mm256_set_epi32(k7,k6,k5,k4,k3,k2,k1,k0);
    __m256i kj1_p1_vec = _mm256_add_epi32(_mm256_set_epi32(0,k7,k6,k5,k4,k3,k2,k1),one_vec);
    __m256i kj1_p4_vec = _mm256_add_epi32(_mm256_set_epi32(0,k7,k6,k5,k4,k3,k2,k1), _mm256_set1_epi32(4));
    __m256i kj2_p1_vec =  _mm256_add_epi32(_mm256_set_epi32(0,0,k7,k6,k5,k4,k3,k2),one_vec);
    __m256i xo_vec = _mm256_set_epi32(a[7].xo,a[6].xo,a[5].xo,a[4].xo,a[3].xo,a[2].xo,a[1].xo,a[0].xo);
    __m256i new_xo_vec = _mm256_add_epi32(xo_vec, _mm256_slli_epi32(_mm256_sub_epi32(k_vec, ak_vec),2));
    __m256i xo_j1_vec = _mm256_and_si256(_mm256_permutevar8x32_epi32(new_xo_vec, shift1_mask),_mm256_set_epi32(0,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff));
    __m256i xo_j2_vec = _mm256_and_si256(_mm256_permutevar8x32_epi32(new_xo_vec, shift1_mask),_mm256_set_epi32(0,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff));
    a[0].xo += (k0 - a[0].k) << 2;
    a[0].k = k0;
    a[1].xo += (k1 - a[1].k) << 2;
    a[1].k = k1;
    a[7].xo += (k7 - a[7].k) << 2;
    a[7].k = k7;
    a[6].xo += (k6 - a[6].k) << 2;
    a[6].k = k6;

    // wfa_next B里应该就是分数+1的节点
    kv_resize(gwf_diag_t, km, *B, B->n + 40);
    b = &B->a[B->n];
    b[0].vd = a[0].vd - 1; //b[0].d == a[0].d-1
    b[0].xo = a[0].xo + 2; // 2 == 1<<1 给反对角线+1，主对角线-1的时候反对角线会+1
    b[0].k = a[0].k + 1; //纵坐标+1，相当于对角线a0右移
    b[0].t = a[0].t;
    b[1].vd = a[0].vd;
    b[1].xo = a[0].k > a[1].k? a[0].xo + 4 : a[1].xo + 2; //如果a0对角线更靠右，则b1对角线为a0右移两位
    b[1].t  =   a[0].k > a[1].k? a[0].t : a[1].t;
    b[1].k  = ( a[0].k > a[1].k? a[0].k : a[1].k) + 1;

    __m256i mask1 = _mm256_cmpgt_epi32(k_vec,kj1_p1_vec);
    __m256i x_res1 = _mm256_blendv_epi8(_mm256_add_epi32(new_xo_vec, _mm256_set1_epi32(2)),
                                        _mm256_add_epi32(xo_j1_vec,_mm256_set1_epi32(4)), mask1);
//    __m256i t_res1 = _mm256_blendv_epi8();
    __m256i k_res1 = _mm256_blendv_epi8(k_vec,kj1_p1_vec,mask1);
    __m256i mask2 = _mm256_cmpgt_epi32(k_res1,kj2_p1_vec);
    __m256i x_res2 = _mm256_blendv_epi8(x_res1, _mm256_add_epi32(xo_j2_vec, _mm256_set1_epi32(2)),mask2);
    __m256i k_res2 = _mm256_blendv_epi8(k_res1, _mm256_add_epi32(kj2_p1_vec, _mm256_set1_epi32(1)),mask2);
    int32_t  x_res[8], k_res[8];
    _mm256_storeu_si256((__m256i *) x_res, x_res2);
    _mm256_storeu_si256((__m256i *) k_res, k_res2);

    b[2].vd = a[1].vd, b[2].k = k_res[0], b[2].xo = x_res[0];
    b[3].vd = a[2].vd, b[3].k = k_res[1], b[3].xo = x_res[1];
    b[4].vd = a[3].vd, b[4].k = k_res[2], b[4].xo = x_res[2];
    b[5].vd = a[4].vd, b[5].k = k_res[3], b[5].xo = x_res[3];
    b[6].vd = a[5].vd, b[6].k = k_res[4], b[6].xo = x_res[4];
    b[7].vd = a[6].vd, b[7].k = k_res[5], b[7].xo = x_res[5];
    b[8].vd = a[7].vd;
    b[8].xo = a[6].k > a[7].k + 1? a[6].xo + 2 : a[7].xo + 4;
    b[8].t  = a[6].k > a[7].k + 1? a[6].t : a[7].t;
    b[8].k  = a[6].k > a[7].k + 1? a[6].k : a[7].k + 1;
    b[9].vd = a[7].vd + 1;
    b[9].xo = a[7].xo + 2;
    b[9].t  = a[7].t;
    b[9].k  = a[7].k;

    // drop out-of-bound cells
    for (j = 0; j < n; ++j) {
        gwf_diag_t *p = &a[j];
        if (p->k == vl - 1 || (int32_t)p->vd - GWF_DIAG_SHIFT + p->k == ql - 1)
            p->xo |= 1, *kdq_pushp(gwf_diag_t, A) = *p; //将所有对角线a中的边缘节点设为乱序，压入A中
    }
    for (j = 0, m = 0; j < n + 2; ++j) {//遍历对角线b
        gwf_diag_t *p = &b[j];
        int32_t d = (int32_t)p->vd - GWF_DIAG_SHIFT;
        if (d + p->k < ql && p->k < vl) {//如果是矩阵内部节点
            b[m++] = *p;//按顺序存储回b
        } else if (p->k == vl) {//如果是边缘节点保存到临时对角线向量q
            gwf_intv_t *q; //
            kv_pushp(gwf_intv_t, km, *tmp_intv, &q);
            q->vd0 = gwf_gen_vd(v, d), q->vd1 = q->vd0 + 1;//对角线a的节点id->v和对角线b的d生成vd0，向下+1生成vd1
        }
    }
//    B->n += m;//441行的m是一个计数器
//    gettimeofday(&en_p_expand, NULL);
//    total_p_expand_time += (double)(en_p_expand.tv_sec - st_p_expand.tv_sec) + (double)(en_p_expand.tv_usec - st_p_expand.tv_usec) / 1000000.0;
}
static gwf_diag_t *gwf_ed_extend_vec(gwf_edbuf_t *buf, const gwf_graph_t *g, int32_t ql, const char *q, int32_t v1, uint32_t max_lag, int32_t traceback,
                                     int32_t *end_v, int32_t *end_off, int32_t *end_tb, int32_t *n_a_, gwf_diag_t *a) {

    int32_t i, x, n = *n_a_, do_dedup = 1;
    kdq_t(gwf_diag_t) *A;
    gwf_diag_v B = {0, 0, 0};
    gwf_diag_t *b;

    *end_v = *end_off = *end_tb = -1;
    buf->tmp.n = 0;
    gwf_set64_clear(buf->ha); // hash table $h to avoid visiting a vertex twice
    for (i = 0, x = 1; i < 32; ++i, x <<= 1)
        if (x >= n) break;
    if (i < 4) i = 4;
    A = kdq_init2(gwf_diag_t, buf->km, i); // $A is a queue
    kv_resize(gwf_diag_t, buf->km, B, n * 2);
#if 1 // unoptimized version without calling gwf_ed_extend_batch() at all. The final result will be the same.
    A->count = n;
    memcpy(A->a, a, n * sizeof(*a));
#else // optimized for long vertices.
    struct timeval st_p_intfinding, en_p_intfinding;
//    gettimeofday(&st_p_intfinding, NULL);

    int pass = 0;
    for(int cnt = 0; cnt+16<=n; cnt = cnt +8) {
//        printf("debugfinder_pcc\n");
        __m256i a1_vec = _mm256_set_epi32((int32_t) a[cnt].vd, (int32_t) a[cnt + 1].vd,
                                          (int32_t) a[cnt + 2].vd, (int32_t) a[cnt + 3].vd,
                                          (int32_t) a[cnt + 4].vd, (int32_t) a[cnt + 5].vd,
                                          (int32_t) a[cnt + 6].vd, (int32_t) a[cnt + 7].vd);
        __m256i *a1_p = &a1_vec;
        __m256i a2_vec = _mm256_set_epi32((int32_t) a[cnt + 8].vd, (int32_t) a[cnt + 9].vd,
                                          (int32_t) a[cnt + 10].vd, (int32_t) a[cnt + 11].vd,
                                          (int32_t) a[cnt + 12].vd, (int32_t) a[cnt + 13].vd,
                                          (int32_t) a[cnt + 14].vd, (int32_t) a[cnt + 15].vd);
        __m256i incre_mask = _mm256_add_epi32(a1_vec, a2_vec);
        int all_ones = _mm256_movemask_epi8(incre_mask);
        if (all_ones == 0xffffffff) {
            gwf_ed_extend_batch_vec(buf->km, g, ql, q, 8, &a[cnt], &B, A, &buf->tmp);
        } else{
            pass = cnt;
            break;
        }
    }
    for (x = pass, i = x+1; i <= n; ++i) {
        if (i == n || a[i].vd != a[i - 1].vd + 1) {
            gwf_ed_extend_batch(buf->km, g, ql, q, i - x, &a[x], &B, A, &buf->tmp);
            x = i;
        }
    }
//    mkae(&en_p_intfinding, NULL);
//    total_p_intfinding_time += (double)(en_p_intfinding.tv_sec - st_p_intfinding.tv_sec) + (double)(en_p_intfinding.tv_usec - st_p_intfinding.tv_usec) / 1000000.0;

//    for (x = 0, i = 1; i <= n; ++i) {
//        //n是可扩展元素个数
//        if (i == n || a[i].vd != a[i-1].vd + 1) {//如果没有可扩展的元素了||下一条对角线不是当前对角线+1
//            gwf_ed_extend_batch_vec2(buf->km, g, ql, q, i - x, &a[x], &B, A, &buf->tmp); //expand创造一个对角线数组b
////            gwf_ed_extend_batch(buf->km, g, ql, q, i - x, &a[x], &B, A, &buf->tmp);
//            printf("pcc=========== x:%d, i:%d\n",x,i);
//            x = i;//expand结束从i继续进行循环直到都expand为止
//        }
//    }
    if (kdq_size(A) == 0) do_dedup = 0;
#endif
    kfree(buf->km, a); // $a is not used as it has been copied to $A
    struct timeval st_p_extend, en_p_extend;
    while (kdq_size(A) >= 8) {
//        gettimeofday(&st_p_extend, NULL);
//        printf("kdq size(vec): %d\n",kdq_size(A));
        gwf_diag_t temp[8];

        for (int i = 0; i < 8; ++i) {
            temp[i] = *kdq_shift(gwf_diag_t, A);
//            printf("%ld\n", temp[i].vd);
        }
        // Load 8 elements from the queue A
        __m256i xo_vector = _mm256_set_epi32(
                (u_int32_t) (temp[7].xo),
                (u_int32_t) (temp[6].xo),
                (u_int32_t) (temp[5].xo),
                (u_int32_t) (temp[4].xo),
                (u_int32_t) (temp[3].xo),
                (u_int32_t) (temp[2].xo),
                (u_int32_t) (temp[1].xo),
                (u_int32_t) (temp[0].xo)
        );
        //__m256i ooo_vector = _mm256_set_epi32((temp[7].xo&1), (temp[6].xo&1), (temp[5].xo&1),(temp[4].xo&1),(temp[3].xo&1),(temp[2].xo&1),(temp[1].xo&1),(temp[0].xo&1));
        __m256i ooo_vector = _mm256_set1_epi32(1);
        __m256i v_vector = _mm256_set_epi32(
                (u_int32_t) (temp[7].vd >> 32),
                (u_int32_t) (temp[6].vd >> 32),
                (u_int32_t) (temp[5].vd >> 32),
                (u_int32_t) (temp[4].vd >> 32),
                (u_int32_t) (temp[3].vd >> 32),
                (u_int32_t) (temp[2].vd >> 32),
                (u_int32_t) (temp[1].vd >> 32),
                (u_int32_t) (temp[0].vd >> 32)
        );
        __m256i d_vector = _mm256_set_epi32(
                ((int32_t)temp[7].vd - GWF_DIAG_SHIFT),
                ((int32_t)temp[6].vd - GWF_DIAG_SHIFT),
                ((int32_t)temp[5].vd - GWF_DIAG_SHIFT),
                ((int32_t)temp[4].vd - GWF_DIAG_SHIFT),
                ((int32_t)temp[3].vd - GWF_DIAG_SHIFT),
                ((int32_t)temp[2].vd - GWF_DIAG_SHIFT),
                ((int32_t)temp[1].vd - GWF_DIAG_SHIFT),
                ((int32_t)temp[0].vd - GWF_DIAG_SHIFT)
        );
        __m256i tmpk_vector = _mm256_set_epi32(//t.k
                (int32_t) (temp[7].k),
                (int32_t) (temp[6].k),
                (int32_t) (temp[5].k),
                (int32_t) (temp[4].k),
                (int32_t) (temp[3].k),
                (int32_t) (temp[2].k),
                (int32_t) (temp[1].k),
                (int32_t) (temp[0].k)
        );
        __m256i ql_vector = _mm256_set1_epi32(ql);
//        );
        __m256i k_vector = _mm256_set_epi32(//after extending k
                (gwf_extend1_vec(((int32_t)temp[7].vd - GWF_DIAG_SHIFT), temp[7].k, ((int32_t)g->len[temp[7].vd >> 32]),
                             g->seq[(int32_t)(temp[7].vd >> 32)], ql, q)),
                (gwf_extend1_vec(((int32_t)temp[6].vd - GWF_DIAG_SHIFT), temp[6].k, ((int32_t)g->len[temp[6].vd >> 32]),
                             g->seq[(int32_t)(temp[6].vd >> 32)], ql, q)),
                (gwf_extend1_vec(((int32_t)temp[5].vd - GWF_DIAG_SHIFT), temp[5].k, ((int32_t)g->len[temp[5].vd >> 32]),
                             g->seq[(int32_t)(temp[5].vd >> 32)], ql, q)),
                (gwf_extend1_vec(((int32_t)temp[4].vd - GWF_DIAG_SHIFT), temp[4].k, ((int32_t)g->len[temp[4].vd >> 32]),
                             g->seq[(int32_t)(temp[4].vd >> 32)], ql, q)),
                (gwf_extend1_vec(((int32_t)temp[3].vd - GWF_DIAG_SHIFT), temp[3].k, ((int32_t)g->len[temp[3].vd >> 32]),
                             g->seq[(int32_t)(temp[3].vd >> 32)], ql, q)),
                (gwf_extend1_vec(((int32_t)temp[2].vd - GWF_DIAG_SHIFT), temp[2].k, ((int32_t)g->len[temp[2].vd >> 32]),
                             g->seq[(int32_t)(temp[2].vd >> 32)], ql, q)),
                (gwf_extend1_vec(((int32_t)temp[1].vd - GWF_DIAG_SHIFT), temp[1].k, ((int32_t)g->len[temp[1].vd >> 32]),
                             g->seq[(int32_t)(temp[1].vd >> 32)], ql, q)),
                (gwf_extend1_vec(((int32_t)temp[0].vd - GWF_DIAG_SHIFT), temp[0].k, ((int32_t)g->len[temp[0].vd >> 32]),
                             g->seq[(int32_t)(temp[0].vd >> 32)], ql, q))
        );
        __m256i vl_vector = _mm256_set_epi32(
                (g->len[(int32_t)(temp[7].vd >> 32)]),
                (g->len[(int32_t)(temp[6].vd >> 32)]),
                (g->len[(int32_t)(temp[5].vd >> 32)]),
                (g->len[(int32_t)(temp[4].vd >> 32)]),
                (g->len[(int32_t)(temp[3].vd >> 32)]),
                (g->len[(int32_t)(temp[2].vd >> 32)]),
                (g->len[(int32_t)(temp[1].vd >> 32)]),
                (g->len[(int32_t)(temp[0].vd >> 32)])
        );
        __m256i t_vector = _mm256_set_epi32(
                (int32_t) (temp[7].t),
                (int32_t) (temp[6].t),
                (int32_t) (temp[5].t),
                (int32_t) (temp[4].t),
                (int32_t) (temp[3].t),
                (int32_t) (temp[2].t),
                (int32_t) (temp[1].t),
                (int32_t) (temp[0].t)
        );
//        printf("temp.t = %d\n", temp[0].t);
        __m256i i_vector = _mm256_add_epi32(k_vector, d_vector);
        // Right shift ooo_vector by 1
        __m256i shifted_xo = _mm256_srli_epi32(xo_vector, 1);
        // Calculate (k - tmpk_vector) and left shift by 1
        __m256i diff = _mm256_sub_epi32(k_vector, tmpk_vector);
        __m256i shifted_diff = _mm256_slli_epi32(diff, 1);
        // Add shifted_ooo and shifted_diff, store result in x_vector
        __m256i x_vector = _mm256_add_epi32(shifted_xo, shifted_diff);
        __m256i ll_x_v = _mm256_slli_epi32(x_vector, 1);
        __m256i new_xo_vector = _mm256_or_si256(ll_x_v, ooo_vector);
        __m256i d_vector_offset = _mm256_add_epi32(d_vector, _mm256_set1_epi32(GWF_DIAG_SHIFT));

        //Branch
        __m256i one_vector = _mm256_set1_epi32(1);
        __m256i kplus1_vector = _mm256_add_epi32(k_vector, one_vector);
        __m256i iplus1_vector = _mm256_add_epi32(i_vector, one_vector);
        __m256i v1_vector = _mm256_set1_epi32(v1);

        //outlayer conditions
        __m256i mask = _mm256_and_si256(_mm256_cmpgt_epi32(vl_vector, kplus1_vector),
                                        _mm256_cmpgt_epi32(ql_vector, iplus1_vector));

        __m256i not_mask = _mm256_xor_si256(mask, _mm256_set1_epi32(0xFFFFFFFF));
        int mask_bits = _mm256_movemask_epi8(mask);
//        printf("mask_bits: %x\n", mask_bits);
        int n_mask_bits = _mm256_movemask_epi8(not_mask);
        uint32_t q1_num = _mm_popcnt_u32(mask_bits) / 4;
//        printf("q1_num: %d\n", q1_num);
        uint32_t q2_num = _mm_popcnt_u32(n_mask_bits) / 4;

        //q1 elements
        __m256i q1_vv = _mm256_and_si256(mask, v_vector);
        __m256i q1_vd = _mm256_and_si256(mask, d_vector_offset);
        __m256i q1_vdminus1 = _mm256_and_si256(mask, _mm256_sub_epi32(d_vector_offset, one_vector));
        __m256i q1_vdplus1 = _mm256_and_si256(mask, _mm256_add_epi32(d_vector_offset, one_vector));
        __m256i q1_vk = _mm256_and_si256(mask, k_vector);
        __m256i q1_vkplus1 = _mm256_and_si256(mask, _mm256_add_epi32(k_vector, one_vector));
        __m256i q1_vx = _mm256_and_si256(mask, new_xo_vector);
        __m256i q1_vxplus2 = _mm256_and_si256(mask, _mm256_add_epi32(new_xo_vector, _mm256_set1_epi32(2)));
        __m256i q1_vxplus4 = _mm256_and_si256(mask, _mm256_add_epi32(new_xo_vector, _mm256_set1_epi32(4)));
        __m256i q1_vt = _mm256_and_si256(mask, t_vector);
//        printf("bugfinder1_pcc\n");

        //q2 elements
        __m256i q2_vv = _mm256_and_si256(not_mask, v_vector);
//        __m256i q2_vd = _mm256_and_si256(not_mask, d_vector_offset);
        __m256i q2_vd = _mm256_and_si256(not_mask, d_vector);
        __m256i q2_vdminus1 = _mm256_and_si256(not_mask, _mm256_sub_epi32(d_vector, one_vector));
        __m256i q2_vdplus1 = _mm256_and_si256(not_mask, _mm256_add_epi32(d_vector, one_vector));
        __m256i q2_vk = _mm256_and_si256(not_mask, k_vector);
        __m256i q2_vkplus1 = _mm256_and_si256(not_mask, _mm256_add_epi32(k_vector, one_vector));
//        __m256i q2_vx = _mm256_and_si256(not_mask, new_xo_vector);
        __m256i q2_vx0 = _mm256_and_si256(not_mask, x_vector);
//        __m256i q2_vxplus2 = _mm256_and_si256(not_mask, _mm256_add_epi32(new_xo_vector, _mm256_set1_epi32(2)));
//        __m256i q2_vxplus4 = _mm256_and_si256(not_mask, _mm256_add_epi32(new_xo_vector, _mm256_set1_epi32(4)));
        __m256i q2_vt = _mm256_and_si256(not_mask, t_vector);
        __m256i q2_vi = _mm256_and_si256(not_mask, i_vector);
        __m256i q2_vvl = _mm256_and_si256(not_mask, vl_vector);
//        printf("bugfinder2_pcc\n");

//        alignas(32) int q1_arr[8]; // AVX2向量寄存器大小为256位，即32字节，因此需要对齐
//        alignas(32) int q2_arr[8];
        uint32_t q1_v_arr[8], q1_xo_arr[8], q1_xop2_arr[8], q1_xop4_arr[8];
        int32_t q1_mask_arr[8], q1_k_arr[8], q1_kp1_arr[8], q1_d_arr[8], q1_dm1_arr[8], q1_dp1_arr[8], q1_t_arr[8];
        _mm256_storeu_si256((__m256i *) q1_v_arr, q1_vv);
        _mm256_storeu_si256((__m256i *) q1_d_arr, q1_vd);
        _mm256_storeu_si256((__m256i *) q1_mask_arr, mask);
        _mm256_storeu_si256((__m256i *) q1_dp1_arr, q1_vdplus1);
        _mm256_storeu_si256((__m256i *) q1_dm1_arr, q1_vdminus1);
        _mm256_storeu_si256((__m256i *) q1_k_arr, q1_vk);
        _mm256_storeu_si256((__m256i *) q1_kp1_arr, q1_vkplus1);
        _mm256_storeu_si256((__m256i *) q1_xo_arr, q1_vx);
        _mm256_storeu_si256((__m256i *) q1_xop2_arr, q1_vxplus2);
        _mm256_storeu_si256((__m256i *) q1_xop4_arr, q1_vxplus4);
        _mm256_storeu_si256((__m256i *) q1_t_arr, q1_vt);
//        printf("bugfinder3_pcc\n");
//
//        for (int i = 0; i < q1_num; i++) {
//            printf("%d ", q1_d_arr[i]);
//        }
//        printf("\n");
        //three cells around push
        gwf_diag_push_vec(buf->km, q1_num, &B, q1_v_arr, q1_dm1_arr, q1_kp1_arr, q1_xop2_arr, q1_t_arr, q1_mask_arr);
        gwf_diag_push_vec(buf->km, q1_num, &B, q1_v_arr, q1_d_arr, q1_kp1_arr, q1_xop4_arr, q1_t_arr, q1_mask_arr);
        gwf_diag_push_vec(buf->km, q1_num, &B, q1_v_arr, q1_dp1_arr, q1_k_arr, q1_xop2_arr, q1_t_arr, q1_mask_arr);

        uint32_t q2_v_arr[8], q2_x0_arr[8], q2_xop2_arr[8], q2_xop4_arr[8];
        int32_t q2_mask_arr[8],q2_vl_arr[8], q2_i_arr[8], q2_k_arr[8], q2_kp1_arr[8], q2_d_arr[8], q2_dm1_arr[8], q2_dp1_arr[8], q2_t_arr[8];
        _mm256_storeu_si256((__m256i *) q2_mask_arr, not_mask);
        _mm256_storeu_si256((__m256i *) q2_v_arr, q2_vv);
        _mm256_storeu_si256((__m256i *) q2_d_arr, q2_vd);
        _mm256_storeu_si256((__m256i *) q2_dp1_arr, q2_vdplus1);
        _mm256_storeu_si256((__m256i *) q2_dm1_arr, q2_vdminus1);
        _mm256_storeu_si256((__m256i *) q2_k_arr, q2_vk);
        _mm256_storeu_si256((__m256i *) q2_kp1_arr, q2_vkplus1);
        _mm256_storeu_si256((__m256i *) q2_x0_arr, q2_vx0);
//        _mm256_storeu_si256((__m256i *) q2_xop2_arr, q2_vxplus2);
//        _mm256_storeu_si256((__m256i *) q2_xop4_arr, q2_vxplus4);
        _mm256_storeu_si256((__m256i *) q2_t_arr, q2_vt);
        _mm256_storeu_si256((__m256i *) q2_i_arr, q2_vi);
        _mm256_storeu_si256((__m256i *) q2_vl_arr, q2_vvl);

//        gettimeofday(&en_p_extend, NULL);
//        total_p_extend_time += (double)(en_p_extend.tv_sec - st_p_extend.tv_sec) + (double)(en_p_extend.tv_usec - st_p_extend.tv_usec) / 1000000.0;

//        printf("bugfinder4_pcc\n");
//        printf("q2d_arr\n");
//        for (int i = 0; i < q1_num; i++) {
//            printf("%d ", q2_d_arr[i]);
//        }
//        printf("\n");

//        printf("kdq size(vec)__1: %d\n",kdq_size(A));
//        printf("q2_num: %d", q2_num);
        for (int32_t c = 0; c < 8; ++c) {
            if (q2_mask_arr[c] == 0) continue;
            if (q2_i_arr[c] + 1 < ql) { // k + 1 == g->len[v]; reaching the end of the vertex but not the end of query
//                printf("kdq size(vec)__2: %d\n",kdq_size(A));
                int32_t ov = g->aux[q2_v_arr[c]] >> 32, nv = (int32_t) g->aux[q2_v_arr[c]], j, n_ext = 0, tw = -1;
                gwf_intv_t *p;
                kv_pushp(gwf_intv_t, buf->km, buf->tmp, &p);
                p->vd0 = gwf_gen_vd(q2_v_arr[c], q2_d_arr[c]), p->vd1 = p->vd0 + 1;
                if (traceback) tw = gwf_trace_push(buf->km, &buf->t, q2_v_arr[c], q2_t_arr[c], buf->ht);
                for (j = 0; j < nv; ++j) { // traverse $v's neighbors
                    uint32_t w = (uint32_t) g->arc[ov + j].a; // $w is next to $v
                    int32_t ol = g->arc[ov + j].o;
                    int absent;
                    gwf_set64_put(buf->ha, (uint64_t) w << 32 | (q2_i_arr[c] + 1),
                                  &absent); // test if ($w,$i) has been visited
                    if (q[q2_i_arr[c] + 1] == g->seq[w][ol]) { // can be extended to the next vertex without a mismatch
                        ++n_ext;
                        if (absent) {
                            gwf_diag_t *p;
                            p = kdq_pushp(gwf_diag_t, A);
                            p->vd = gwf_gen_vd(w, q2_i_arr[c] + 1 - ol), p->k = ol, p->xo =
                                    (q2_x0_arr[c] + 2) << 1 | 1, p->t = tw;
                        }
                    } else if (absent) {
//                        printf("k: %d, i: %d, ol : %d\n", q2_k_arr[c], q2_i_arr[c], ol);
                        gwf_diag_push(buf->km, &B, w, q2_i_arr[c] - ol, ol, q2_x0_arr[c] + 1, 1, tw);
                        gwf_diag_push(buf->km, &B, w, q2_i_arr[c] + 1 - ol, ol, q2_x0_arr[c] + 2, 1, tw);
                    }
                }
                if (nv == 0 ||
                    n_ext != nv) // add an insertion to the target; this *might* cause a duplicate in corner cases
                    gwf_diag_push(buf->km, &B, q2_v_arr[c], q2_dp1_arr[c], q2_k_arr[c], q2_x0_arr[c] + 1, 1,
                                  q2_t_arr[c]);
            } else if (v1 < 0 || (q2_v_arr[c] == v1 && q2_k_arr[c] + 1 == q2_vl_arr[c])) { // i + 1 == ql
//                printf("kdq size(vec)__2: %d\n",kdq_size(A));
                *end_v = q2_v_arr[c], *end_off = q2_k_arr[c], *end_tb = q2_t_arr[c], *n_a_ = 0;
                kdq_destroy(gwf_diag_t, A);
                kfree(buf->km, B.a);
                return 0;
            } else if (q2_k_arr[c] + 1 <
                       q2_vl_arr[c]) { // i + 1 == ql; reaching the end of the query but not the end of the vertex
//                printf("kdq size(vec)__3: %d\n",kdq_size(A));
                gwf_diag_push(buf->km, &B, q2_v_arr[c], q2_dm1_arr[c], q2_kp1_arr[c], q2_x0_arr[c] + 1, 1,
                              q2_t_arr[c]); // add an deletion; this *might* case a duplicate in corner cases
            } else if (q2_v_arr[c] != v1) { // i + 1 == ql && k + 1 == g->len[v]; not reaching the last vertex $v1
//                printf("kdq size(vec)__4: %d\n",kdq_size(A));
                int32_t ov = g->aux[q2_v_arr[c]] >> 32, nv = (int32_t) g->aux[q2_v_arr[c]], j, tw = -1;
                if (traceback) tw = gwf_trace_push(buf->km, &buf->t, q2_v_arr[c], q2_t_arr[c], buf->ht);
                for (j = 0; j < nv; ++j) {
                    uint32_t w = (uint32_t) g->arc[ov + j].a;
                    int32_t ol = g->arc[ov + j].o;
                    gwf_diag_push(buf->km, &B, w, i - ol, ol, q2_x0_arr[c] + 1, 1,
                                  tw); // deleting the first base on the next vertex
                }
            } else
                assert(0); // should never come here
        }
    }

    while (kdq_size(A)) {
        gwf_diag_t t;
        uint32_t x0;
        int32_t ooo, v, d, k, i, vl;
//        printf("kdq size: %d\n",kdq_size(A));

        t = *kdq_shift(gwf_diag_t, A);
        ooo = t.xo&1, v = t.vd >> 32; // vertex
        d = (int32_t)t.vd - GWF_DIAG_SHIFT; // diagonal
        k = t.k; // wavefront position on the vertex
        vl = g->len[v]; // $vl is the vertex length
        k = gwf_extend1_vec(d, k, vl, g->seq[v], ql, q);
//        k = gwf_extend1_vec(d, k, vl, g->seq[v], ql, q);
        i = k + d; // query position
        x0 = (t.xo >> 1) + ((k - t.k) << 1); // current anti diagonal

        if (k + 1 < vl && i + 1 < ql) { // the most common case: the wavefront is in the middle
            int32_t push1 = 1, push2 = 1;
            if (B.n >= 2) push1 = gwf_diag_update(&B.a[B.n - 2], v, d-1, k+1, x0 + 1, ooo, t.t);
            if (B.n >= 1) push2 = gwf_diag_update(&B.a[B.n - 1], v, d,   k+1, x0 + 2, ooo, t.t);
            if (push1)          gwf_diag_push(buf->km, &B, v, d-1, k+1, x0 + 1, 1, t.t);
            if (push2 || push1) gwf_diag_push(buf->km, &B, v, d,   k+1, x0 + 2, 1, t.t);
            gwf_diag_push(buf->km, &B, v, d+1, k, x0 + 1, ooo, t.t);
        } else if (i + 1 < ql) { // k + 1 == g->len[v]; reaching the end of the vertex but not the end of query
            int32_t ov = g->aux[v]>>32, nv = (int32_t)g->aux[v], j, n_ext = 0, tw = -1;
            gwf_intv_t *p;
            kv_pushp(gwf_intv_t, buf->km, buf->tmp, &p);
            p->vd0 = gwf_gen_vd(v, d), p->vd1 = p->vd0 + 1;
            if (traceback) tw = gwf_trace_push(buf->km, &buf->t, v, t.t, buf->ht);
            for (j = 0; j < nv; ++j) { // traverse $v's neighbors
                uint32_t w = (uint32_t)g->arc[ov + j].a; // $w is next to $v
                int32_t ol = g->arc[ov + j].o;
                int absent;
                gwf_set64_put(buf->ha, (uint64_t)w<<32 | (i + 1), &absent); // test if ($w,$i) has been visited
                if (q[i + 1] == g->seq[w][ol]) { // can be extended to the next vertex without a mismatch
                    ++n_ext;
                    if (absent) {
                        gwf_diag_t *p;
                        p = kdq_pushp(gwf_diag_t, A);
                        p->vd = gwf_gen_vd(w, i+1-ol), p->k = ol, p->xo = (x0+2)<<1 | 1, p->t = tw;
                    }
                } else if (absent) {
                    gwf_diag_push(buf->km, &B, w, i-ol,   ol, x0 + 1, 1, tw);
                    gwf_diag_push(buf->km, &B, w, i+1-ol, ol, x0 + 2, 1, tw);
                }
            }
            if (nv == 0 || n_ext != nv) // add an insertion to the target; this *might* cause a duplicate in corner cases
                gwf_diag_push(buf->km, &B, v, d+1, k, x0 + 1, 1, t.t);
        } else if (v1 < 0 || (v == v1 && k + 1 == vl)) { // i + 1 == ql
            *end_v = v, *end_off = k, *end_tb = t.t, *n_a_ = 0;
            kdq_destroy(gwf_diag_t, A);
            kfree(buf->km, B.a);
            return 0;
        } else if (k + 1 < vl) { // i + 1 == ql; reaching the end of the query but not the end of the vertex
            gwf_diag_push(buf->km, &B, v, d-1, k+1, x0 + 1, ooo, t.t); // add an deletion; this *might* case a duplicate in corner cases
        } else if (v != v1) { // i + 1 == ql && k + 1 == g->len[v]; not reaching the last vertex $v1
            int32_t ov = g->aux[v]>>32, nv = (int32_t)g->aux[v], j, tw = -1;
            if (traceback) tw = gwf_trace_push(buf->km, &buf->t, v, t.t, buf->ht);
            for (j = 0; j < nv; ++j) {
                uint32_t w = (uint32_t)g->arc[ov + j].a;
                int32_t ol = g->arc[ov + j].o;
                gwf_diag_push(buf->km, &B, w, i-ol, ol, x0 + 1, 1, tw); // deleting the first base on the next vertex
            }
        } else assert(0); // should never come here
    }
//    printf("B__size: %d\n", B.n);
    kdq_destroy(gwf_diag_t, A);
    *n_a_ = n = B.n, b = B.a;

    if (do_dedup) *n_a_ = n = gwf_dedup(buf, n, b);
    if (max_lag > 0) *n_a_ = n = gwf_prune(n, b, max_lag);
//    printf("B__size: %d\n", n);
//    printf("Prune begin:\n");
//    printf("Prune end:\n");
    return b;
}
// wfa_extend and wfa_next combined
static gwf_diag_t *gwf_ed_extend(gwf_edbuf_t *buf, const gwf_graph_t *g, int32_t ql, const char *q, int32_t v1, uint32_t max_lag, int32_t traceback,
								 int32_t *end_v, int32_t *end_off, int32_t *end_tb, int32_t *n_a_, gwf_diag_t *a)
{
	int32_t i, x, n = *n_a_, do_dedup = 1;
	kdq_t(gwf_diag_t) *A;
	gwf_diag_v B = {0,0,0};
	gwf_diag_t *b;

	*end_v = *end_off = *end_tb = -1;
	buf->tmp.n = 0;
	gwf_set64_clear(buf->ha); // hash table $h to avoid visiting a vertex twice
	for (i = 0, x = 1; i < 32; ++i, x <<= 1)
		if (x >= n) break;
	if (i < 4) i = 4;
	A = kdq_init2(gwf_diag_t, buf->km, i); // $A is a queue
	kv_resize(gwf_diag_t, buf->km, B, n * 2);
#if 0 // unoptimized version without calling gwf_ed_extend_batch() at all. The final result will be the same.
	A->count = n;
	memcpy(A->a, a, n * sizeof(*a));
#else // optimized for long vertices.
    struct timeval st_s_intfinding, en_s_intfinding;
//    gettimeofday(&st_s_intfinding, NULL);
	for (x = 0, i = 1; i <= n; ++i) {
		if (i == n || a[i].vd != a[i-1].vd + 1) {
			gwf_ed_extend_batch(buf->km, g, ql, q, i - x, &a[x], &B, A, &buf->tmp);
			x = i;
		}
	}
//    gettimeofday(&en_s_intfinding, NULL);
//    total_s_intfinding_time += (double)(en_s_intfinding.tv_sec - st_s_intfinding.tv_sec) + (double)(en_s_intfinding.tv_usec - st_s_intfinding.tv_usec) / 1000000.0;

    if (kdq_size(A) == 0) do_dedup = 0;
#endif
	kfree(buf->km, a); // $a is not used as it has been copied to $A
    struct timeval st_s_extend, en_s_extend;
	while (kdq_size(A)) {
//        gettimeofday(&st_s_extend, NULL);
		gwf_diag_t t;
		uint32_t x0;
		int32_t ooo, v, d, k, i, vl;

		t = *kdq_shift(gwf_diag_t, A);
		ooo = t.xo&1, v = t.vd >> 32; // vertex
		d = (int32_t)t.vd - GWF_DIAG_SHIFT; // diagonal
		k = t.k; // wavefront position on the vertex
		vl = g->len[v]; // $vl is the vertex length
		k = gwf_extend1(d, k, vl, g->seq[v], ql, q);
		i = k + d; // query position
		x0 = (t.xo >> 1) + ((k - t.k) << 1); // current anti diagonal

		if (k + 1 < vl && i + 1 < ql) { // the most common case: the wavefront is in the middle
			int32_t push1 = 1, push2 = 1;
			if (B.n >= 2) push1 = gwf_diag_update(&B.a[B.n - 2], v, d-1, k+1, x0 + 1, ooo, t.t);
			if (B.n >= 1) push2 = gwf_diag_update(&B.a[B.n - 1], v, d,   k+1, x0 + 2, ooo, t.t);
			if (push1)          gwf_diag_push(buf->km, &B, v, d-1, k+1, x0 + 1, 1, t.t);
			if (push2 || push1) gwf_diag_push(buf->km, &B, v, d,   k+1, x0 + 2, 1, t.t);
			gwf_diag_push(buf->km, &B, v, d+1, k, x0 + 1, ooo, t.t);
//            gettimeofday(&en_s_extend, NULL);
//            total_s_extend_time += (double)(en_s_extend.tv_sec - st_s_extend.tv_sec) + (double)(en_s_extend.tv_usec - st_s_extend.tv_usec) / 1000000.0;
        } else if (i + 1 < ql) { // k + 1 == g->len[v]; reaching the end of the vertex but not the end of query
			int32_t ov = g->aux[v]>>32, nv = (int32_t)g->aux[v], j, n_ext = 0, tw = -1;
			gwf_intv_t *p;
			kv_pushp(gwf_intv_t, buf->km, buf->tmp, &p);
			p->vd0 = gwf_gen_vd(v, d), p->vd1 = p->vd0 + 1;
			if (traceback) tw = gwf_trace_push(buf->km, &buf->t, v, t.t, buf->ht);
			for (j = 0; j < nv; ++j) { // traverse $v's neighbors
				uint32_t w = (uint32_t)g->arc[ov + j].a; // $w is next to $v
				int32_t ol = g->arc[ov + j].o;
				int absent;
				gwf_set64_put(buf->ha, (uint64_t)w<<32 | (i + 1), &absent); // test if ($w,$i) has been visited
				if (q[i + 1] == g->seq[w][ol]) { // can be extended to the next vertex without a mismatch
					++n_ext;
					if (absent) {
						gwf_diag_t *p;
						p = kdq_pushp(gwf_diag_t, A);
						p->vd = gwf_gen_vd(w, i+1-ol), p->k = ol, p->xo = (x0+2)<<1 | 1, p->t = tw;
					}
				} else if (absent) {
					gwf_diag_push(buf->km, &B, w, i-ol,   ol, x0 + 1, 1, tw);
					gwf_diag_push(buf->km, &B, w, i+1-ol, ol, x0 + 2, 1, tw);
				}
			}
			if (nv == 0 || n_ext != nv) // add an insertion to the target; this *might* cause a duplicate in corner cases
				gwf_diag_push(buf->km, &B, v, d+1, k, x0 + 1, 1, t.t);
		} else if (v1 < 0 || (v == v1 && k + 1 == vl)) { // i + 1 == ql
			*end_v = v, *end_off = k, *end_tb = t.t, *n_a_ = 0;
			kdq_destroy(gwf_diag_t, A);
			kfree(buf->km, B.a);
			return 0;
		} else if (k + 1 < vl) { // i + 1 == ql; reaching the end of the query but not the end of the vertex
			gwf_diag_push(buf->km, &B, v, d-1, k+1, x0 + 1, ooo, t.t); // add an deletion; this *might* case a duplicate in corner cases
		} else if (v != v1) { // i + 1 == ql && k + 1 == g->len[v]; not reaching the last vertex $v1
			int32_t ov = g->aux[v]>>32, nv = (int32_t)g->aux[v], j, tw = -1;
			if (traceback) tw = gwf_trace_push(buf->km, &buf->t, v, t.t, buf->ht);
			for (j = 0; j < nv; ++j) {
				uint32_t w = (uint32_t)g->arc[ov + j].a;
				int32_t ol = g->arc[ov + j].o;
				gwf_diag_push(buf->km, &B, w, i-ol, ol, x0 + 1, 1, tw); // deleting the first base on the next vertex
			}
		} else assert(0); // should never come here
	}

	kdq_destroy(gwf_diag_t, A);
	*n_a_ = n = B.n, b = B.a;

	if (do_dedup) *n_a_ = n = gwf_dedup(buf, n, b);
	if (max_lag > 0) *n_a_ = n = gwf_prune(n, b, max_lag);
	return b;
}

static void gwf_traceback(gwf_edbuf_t *buf, int32_t end_v, int32_t end_tb, gwf_path_t *path)
{
	int32_t i = end_tb, n = 1;
	while (i >= 0 && buf->t.a[i].v >= 0)
		++n, i = buf->t.a[i].pre;
	KMALLOC(buf->km, path->v, n);
	i = end_tb, n = 0;
	path->v[n++] = end_v;
	while (i >= 0 && buf->t.a[i].v >= 0)
		path->v[n++] = buf->t.a[i].v, i = buf->t.a[i].pre;
	path->nv = n;
	for (i = 0; i < path->nv>>1; ++i)
		n = path->v[i], path->v[i] = path->v[path->nv - 1 - i], path->v[path->nv - 1 - i] = n;
}

int32_t gwf_ed(void *km, const gwf_graph_t *g, int32_t ql, const char *q, int32_t v0, int32_t v1, uint32_t max_lag, int32_t traceback, gwf_path_t *path, long my_rank, int proc)
{
	int32_t s = 0, n_a = 1, end_tb;
	gwf_diag_t *a;
	gwf_edbuf_t buf;

	memset(&buf, 0, sizeof(buf));
	buf.km = km;
	buf.ha = gwf_set64_init2(km);
	buf.ht = gwf_map64_init2(km);
	kv_resize(gwf_trace_t, km, buf.t, g->n_vtx + 16);
	KCALLOC(km, a, 1);
	a[0].vd = gwf_gen_vd(v0, 0), a[0].k = -1, a[0].xo = 0; // the initial state
	if (traceback) a[0].t = gwf_trace_push(km, &buf.t, -1, -1, buf.ht);
	while (n_a > 0) {
//		a = gwf_ed_extend_vec(&buf, g, ql, q, v1, max_lag, traceback, &path->end_v, &path->end_off, &end_tb, &n_a, a);
        a = gwf_ed_extend(&buf, g, ql, q, v1, max_lag, traceback, &path->end_v, &path->end_off, &end_tb, &n_a, a);
		if (path->end_off >= 0 || n_a == 0) break;
		++s;
#ifdef GWF_DEBUG
		printf("[%s] dist=%d, n=%d, n_intv=%ld, n_tb=%ld, thread %ld, proc%d\n", __func__, s, n_a, buf.intv.n, buf.t.n, my_rank, proc);
#endif
	}
	if (traceback) gwf_traceback(&buf, path->end_v, end_tb, path);
	gwf_set64_destroy(buf.ha);
	gwf_map64_destroy(buf.ht);
	kfree(km, buf.intv.a); kfree(km, buf.tmp.a); kfree(km, buf.swap.a); kfree(km, buf.t.a);
	path->s = path->end_v >= 0? s : -1;
	return path->s; // end_v < 0 could happen if v0 can't reach v1
}
