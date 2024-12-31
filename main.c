#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <zlib.h>
#include <pthread.h>
// #include<sys/time.h> //使用该文件内的struct timeval结构体获取时间信息
#include "gfa.h"
#include "gfa-priv.h"
#include "gwfa.h"
#include "ketopt.h"
#include "kalloc.h"
#include "kseq.h"
#include "timer.h"
#include <mpi.h>
#include "time.h"
#include "global_variants.h"

pthread_mutex_t lock;
pthread_cond_t cond;
int task_index[8];
#define MAX_THREADS 128

long thread_count;
struct parameter{
    string_t *stb;
	string_t *local_stb;
	gwf_path_t *ptb;
    // int thread;
    int n, local_n;
    MPI_Comm comm;
    int my_rank, comm_sz;
};
struct parameter *par;
string_t svb;
long seqs = 0;
long bases = 0; 
size_t slen = 0; 

gwf_graph_t *g;
gfa_t *gfa;
// gwf_path_t *ptb;
int traceback = 0;
uint32_t max_lag = 0;
void *km = 0;

KSEQ_INIT(gzFile, gzread)
kseq_t *ks;

typedef struct
{
	char name[20];
	int time;
	int start;
	int end;
	void *km2;
	gwf_graph_t *g2;

}RaceArg;

gwf_graph_t *gwf_gfa2gwf(const gfa_t *gfa, uint32_t v0)
{
	int32_t i, k;
	gwf_graph_t *g;
	gfa_sub_t *sub;
	sub = gfa_sub_from(0, gfa, v0, 1<<30);
	GFA_CALLOC(g, 1);
	g->n_vtx = sub->n_v;
	g->n_arc = sub->n_a;
	GFA_MALLOC(g->len, g->n_vtx);
	GFA_MALLOC(g->src, g->n_vtx);
	GFA_MALLOC(g->seq, g->n_vtx);
	GFA_MALLOC(g->arc, g->n_arc);
	for (i = k = 0; i < sub->n_v; ++i) {
		uint32_t v = sub->v[i].v, len = gfa->seg[v>>1].len, j;
		const gfa_seg_t *s = &gfa->seg[v>>1];
		g->len[i] = len;
		g->src[i] = v;
		GFA_MALLOC(g->seq[i], len + 1);
		if (v&1) {
			for (j = 0; j < len; ++j)
				g->seq[i][j] = gfa_comp_table[(uint8_t)s->seq[len - j - 1]];
		} else memcpy(g->seq[i], s->seq, len);
		g->seq[i][len] = 0; // null terminated for convenience
		for (j = 0; j < sub->v[i].n; ++j) {
			uint64_t a = sub->a[sub->v[i].off + j];
			g->arc[k].a = (uint64_t)i<<32 | a>>32;
			g->arc[k].o = gfa->arc[(uint32_t)a].ow;
			++k;
		}
		assert(k <= g->n_arc);
	}
	return g;
}

void gwf_free(gwf_graph_t *g)
{
	int32_t i;
	for (i = 0; i < g->n_vtx; ++i) free(g->seq[i]);
	free(g->len); free(g->seq); free(g->arc); free(g->src); free(g);
}

void gwf_graph_print(FILE *fp, const gwf_graph_t *g)
{
	int32_t i;
	for (i = 0; i < g->n_vtx; ++i)
		fprintf(fp, "S\t%d\t%s\tLN:i:%d\n", i, g->seq[i], g->len[i]);
	for (i = 0; i < g->n_arc; ++i)
		fprintf(fp, "L\t%d\t+\t%d\t+\t%dM\n", (uint32_t)(g->arc[i].a>>32), (uint32_t)g->arc[i].a, g->arc[i].o);
}
void* Thread_worker(void* rank);//added
void* Thread_gwf(void* rank);
void Build_mpi_type(int* l_p, char** s_p, char** name_p,MPI_Datatype* input_mpi_t_p);
void Read_seq(int* n_p, int* local_n_p, string_t stb[], int my_rank, int comm_sz, MPI_Comm comm);
void Read_seq_sort(int* n_p, int* local_n_p, string_t stb[], int my_rank, int comm_sz, MPI_Comm comm);
void Allocate(string_t **local_stb_pp, int local_n, MPI_Comm comm);
void Read_vector(string_t local_stb[], int local_n, int n, int my_rank, MPI_Comm comm);
void quickSort(string_t stb[], int low, int high);
int partition(string_t stb[], int low, int high);
void swap(string_t *a, string_t *b);

int main(int argc, char *argv[])
{
	gzFile fp; //zlib压缩文件操作
	// kseq_t *ks;
	ketopt_t o = KETOPT_INIT;
	
	int c, print_graph = 0;//, traceback = 0;
	uint32_t v0 = 0<<1|0; // first segment, forward strand
	char *sname = 0;

	double start,finish,start1,finish1,elapsed1,elapsed2,elapsed3,elapsed4,loc_elapsed,elapsed;

	par =  (struct parameter *) malloc (sizeof(struct parameter));
	par->stb = (string_t*) malloc (100000*sizeof(string_t));
	par->ptb = (gwf_path_t*) malloc (100000*sizeof(gwf_path_t));
	
	long thread;
    pthread_t* thread_handles;
	

	while ((c = ketopt(&o, argc, argv, 1, "ptl:s:", 0)) >= 0) {
		if (c == 'p') print_graph = 1;
		else if (c == 'l') max_lag = atoi(o.arg);
		else if (c == 's') sname = o.arg;
		else if (c == 't') traceback = 1;
	}//解析过程
	if ((!print_graph && argc - o.ind < 2) || (print_graph && argc == o.ind)) {
		fprintf(stderr, "Usage: gwf-test [options] <target.gfa|fa> <query.fa>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -l INT    max lag behind the furthest wavefront; 0 to disable [0]\n");
		fprintf(stderr, "  -s STR    starting segment name [first]\n");
		fprintf(stderr, "  -t        report the alignment path\n");
		fprintf(stderr, "  -p        output GFA in the forward strand\n");
		return 1;
	}

	thread_count = strtol(argv[o.ind+2], NULL, 10);
	MPI_Init(NULL, NULL);
    par->comm = MPI_COMM_WORLD;
    MPI_Comm_size(par->comm, &(par->comm_sz));
    MPI_Comm_rank(par->comm, &(par->my_rank));
    thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t)); 

	GET_TIME(start);
	km = km_init();
	gfa = gfa_read(argv[o.ind]);
	assert(gfa);
	if (sname) {
		int32_t sid;
		sid = gfa_name2id(gfa, sname);
		if (sid < 0) fprintf(stderr, "ERROR: failed to find segment '%s'\n", sname);
		else v0 = sid<<1 | 0; // TODO: also allow to change the orientation
	}
	GET_TIME(finish);
	elapsed1 = finish - start;
	// double t1=(time_end1.tv_sec-time_start1.tv_sec)*1000+(time_end1.tv_usec-time_start1.tv_usec)/1000;  //转成ms表示

	GET_TIME(start);
	g = gwf_gfa2gwf(gfa, v0);
	if (print_graph) {
		gwf_graph_print(stdout, g);
		return 0; // free memory
	}
	gwf_ed_index(km, g);
	GET_TIME(finish);
	elapsed2 = finish - start;
	// double t2=(time_end2.tv_sec-time_start2.tv_sec)*1000+(time_end2.tv_usec-time_start2.tv_usec)/1000;  //转成ms表示

	GET_TIME(start);
	fp = gzopen(argv[o.ind+1], "r");
    printf("pcc1\n");
	assert(fp);
	ks = kseq_init(fp);
	GET_TIME(finish);
	elapsed3 = finish - start;
	// double t3=(time_end3.tv_sec-time_start3.tv_sec)*1000+(time_end3.tv_usec-time_start3.tv_usec)/1000;  //转成ms表示

	
//	Read_seq_sort(&(par->n), &(par->local_n), par->stb, par->my_rank, par->comm_sz, par->comm);
    Read_seq(&(par->n), &(par->local_n), par->stb, par->my_rank, par->comm_sz, par->comm);
    printf("pcc2\n");
#   ifdef DEBUG
    printf("Proc %d > n = %d, local_n = %d\n", par->my_rank, par->n, par->local_n);
#   endif
	Allocate(&(par->local_stb), par->local_n, par->comm);
	Read_vector(par->local_stb, par->local_n, par->n, par->my_rank, par->comm);
	GET_TIME(start);
	start1 = MPI_Wtime();
    for(thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, Thread_gwf, (void*)thread);

    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);

	GET_TIME(finish);
	elapsed4 = finish - start;
	finish1 = MPI_Wtime();
	loc_elapsed = finish1 - start1;
    printf("Proc %d: my elapsed time = %f seconds\n", par->my_rank, loc_elapsed);
	MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, par->comm);
	// printf("seq: %s\n", stb[7].s);
	// printf("length: %ld\n", stb[7].l);
	if (par->my_rank == 0){
		printf("reads: %ld\n", seqs);
		printf("bases: %ld\n", bases);
	}
	
	// double t4=(time_end4.tv_sec-time_start4.tv_sec)*1000+(time_end4.tv_usec-time_start4.tv_usec)/1000;  //转成ms表示
	kseq_destroy(ks);
	gzclose(fp);

	gfa_destroy(gfa);

	gwf_cleanup(km, g);
	gwf_free(g);
	km_destroy(km);
	if (par->my_rank == 0){
		printf("The elapsed time for GFA is %e seconds\n",elapsed1);//
		printf("The elapsed time for gwf process is %e seconds\n",elapsed2);//
		printf("The elapsed time for fa is %e seconds\n",elapsed3);//
		printf("The elapsed time for cp is %e seconds\n",elapsed4);//
	
        printf("Elapsed time = %e\n", elapsed);
	}


	free(thread_handles);
	free(par->stb);
	free(par->ptb);
	MPI_Finalize();
	return 0;
}
void swap(string_t *a, string_t *b) {
    string_t temp = *a;
    *a = *b;
    *b = temp;
}

// 分区函数：以最后一个元素作为基准，将数组分为两部分，小于基准的放在左边，大于基准的放在右边
//int partition(string_t arr[], int low, int high) {
//    int pivot = arr[high].l; // 将最后一个元素设为基准
//    int i = low - 1; // 初始化分区索引，i指向小于基准的最后一个元素
//
//    for (int j = low; j < high; j++) {
//        if (arr[j].l < pivot) {
//            i++; // 扩展小于基准元素的分区
//            swap(&arr[i], &arr[j]); // 交换arr[i]和arr[j]
//        }
//    }
//    swap(&arr[i + 1], &arr[high]); // 将基准元素放到正确的位置
//    return i + 1; // 返回基准元素的索引
//}

int partition(string_t arr[], int low, int high) {
    int pivot = arr[high].l; // 将最后一个元素设为基准
    int i = low - 1; // 初始化分区索引，i指向大于基准的最后一个元素

    for (int j = low; j < high; j++) {
        if (arr[j].l > pivot) { // 注意这里将比较条件颠倒
            i++; // 扩展大于基准元素的分区
            swap(&arr[i], &arr[j]); // 交换arr[i]和arr[j]
        }
    }
    swap(&arr[i + 1], &arr[high]); // 将基准元素放到正确的位置
    return i + 1; // 返回基准元素的索引
}


// 快速排序函数
void quickSort(string_t arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high); // 找到基准元素的位置
        quickSort(arr, low, pi - 1); // 对基准元素左边的子数组进行快速排序
        quickSort(arr, pi + 1, high); // 对基准元素右边的子数组进行快速排序
    }
}

void Read_seq(
      int*      n_p        /* out */, 
      int*      local_n_p  /* out */, 
	  string_t  stb[],
      int       my_rank    /* in  */, 
      int       comm_sz    /* in  */,
      MPI_Comm  comm       /* in  */) {
   
//    if (my_rank == 0) {
	while (kseq_read(ks) >= 0) {
        svb.s = strdup(ks->seq.s);
        svb.name = strdup(ks->name.s);
	    svb.l = strlen(ks->seq.s);
	    stb[seqs] = svb;

	    slen = strlen(ks->name.s);
	    bases += strlen(ks->seq.s);
	    seqs += 1;
	}
//    }
   *n_p = seqs;
   MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
   *local_n_p = *n_p/comm_sz;
}

void Read_seq_sort(
        int*      n_p        /* out */,
        int*      local_n_p  /* out */,
        string_t  stb[],
        int       my_rank    /* in  */,
        int       comm_sz    /* in  */,
        MPI_Comm  comm       /* in  */) {

//    if (my_rank == 0) {
    int myseqs_counter = 0;
    while (kseq_read(ks) >= 0) {
        svb.s = strdup(ks->seq.s);
        svb.name = strdup(ks->name.s);
//        svb.s = ks->seq.s;
//        svb.name = ks->name.s;
        svb.l = strlen(ks->seq.s);
        stb[seqs] = svb;

        slen = strlen(ks->name.s);
        bases += strlen(ks->seq.s);
        seqs += 1;
        myseqs_counter += 1;
    }
//    }
//   *n_p = seqs;
//   MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
//   *local_n_p = *n_p/comm_sz;
//    printf("I am proc %d, my seqs is %d \n",my_rank, myseqs_counter);
    assert(seqs);
    //MPI_Bcast(&seqs, 1, MPI_LONG, 0, comm);

//mpi load balance   //sort by size
    quickSort(stb, 0, seqs - 1);
//mpi load balance

    // 计算余数
    int remainder = seqs % comm_sz;

    // 计算本地序列数
    if (remainder == 0) {
        *local_n_p = seqs / comm_sz;
    } else {
        // 余数分配法
        if (my_rank < remainder) {
            *local_n_p = seqs / comm_sz + 1;
        } else {
            *local_n_p = seqs / comm_sz;
        }
    }
    *n_p = seqs;
    printf("local_n: %d from proc %d\n", par->local_n, par->my_rank);
}

void Allocate(
      string_t **local_stb_pp,  
      int        local_n     /* in  */,
      MPI_Comm   comm        /* in  */) {

   *local_stb_pp = (string_t*) malloc ((local_n+1)*sizeof(string_t));
}

void Build_mpi_type(
	int* l_p,
	char** s_p,
	char** name_p,
	MPI_Datatype* input_mpi_t_p){
  int array_of_blocklengths[3] = {1, bases+1, slen+1};
  MPI_Datatype array_of_types[3] = {MPI_INT, MPI_CHAR, MPI_CHAR, MPI_CHAR};
  MPI_Aint l_addr, s_addr, name_addr;
  MPI_Aint array_of_displacements[3] = {0};

  MPI_Get_address(l_p, &l_addr);
  MPI_Get_address(s_p, &s_addr);
  MPI_Get_address(name_p, &name_addr);
  array_of_displacements[1] = s_addr - l_addr;
  array_of_displacements[2] = name_addr - l_addr;
  MPI_Type_create_struct(3, array_of_blocklengths,array_of_displacements ,array_of_types,
  	input_mpi_t_p);
  MPI_Type_commit(input_mpi_t_p);
}

void Read_vector(
      string_t    local_stb[]   /* out */,
      int       local_n     /* in  */,
      int       n           /* in  */,
      int       my_rank     /* in  */,
      MPI_Comm  comm        /* in  */) {

    string_t* a = NULL;
    int i;
    for (i = 0; i < local_n; i++){
        local_stb[i] = par->stb[par->my_rank*local_n+i];
	}

	MPI_Datatype input_mpi_t;
    Build_mpi_type(&(svb.l),&(svb.s),&(svb.name),&input_mpi_t);
//    if (my_rank == 0) {
//       a = (string_t*)malloc(n*sizeof(string_t));
//       for (i = 0; i < local_n; i++){
//          local_stb[i] = par->stb[i];

//       }
//       MPI_Scatter(a, local_n, input_mpi_t, local_stb, local_n, input_mpi_t, 0,
//          comm);
//       free(a);
//    } else {
//       MPI_Scatter(a, local_n, input_mpi_t, local_stb, local_n, input_mpi_t, 0,
//          comm);
//    }
}
//void Read_vector(
//        string_t    local_stb[]   /* out */,
//        int       local_n     /* in  */,
//        int       n           /* in  */,
//        int       my_rank     /* in  */,
//        MPI_Comm  comm        /* in  */) {
//
//    string_t* a = NULL;
//    int i;
//    for (i = 0; i < local_n; i++){
//        int global_index = my_rank + i * par->comm_sz;
//        local_stb[i] = par->stb[global_index];
//        printf("%d seq from %d proc\n",global_index, my_rank);
//    }
////    int remainder = n % par->comm_sz;
////    int start_index = my_rank * local_n + (my_rank < remainder ? my_rank : remainder);
////    int end_index = start_index + local_n + (my_rank < remainder ? 1 : 0) - 1;
////
////    // 根据处理器的计算能力分配任务索引
////    for (int i = 0; i < local_n; i++) {
////        // 从最小的任务开始取
////        int global_index = start_index + i;
////        if (global_index > end_index)
////            global_index = end_index - (i % (local_n + (my_rank < remainder ? 1 : 0)));
////        local_stb[i] = par->stb[global_index];
////        printf("%d seq from %d proc\n", global_index, my_rank);
////    }
//    MPI_Datatype input_mpi_t;
//    Build_mpi_type(&(svb.l),&(svb.s),&(svb.name),&input_mpi_t);
//}

void* Thread_gwf(void* rank) {
    struct timespec start, end;
    double thread_elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    long my_rank = (long) rank;
    long long i;
    long long my_seqs = par->local_n/thread_count;
    long long my_first_i = my_seqs*my_rank;
    long long my_last_i = my_first_i + my_seqs;
	
    for(i = my_first_i; i < my_last_i; i++) {
		// printf("Now is thread %ld of proc %d, sequence No.%lld: %s\n",my_rank,par->my_rank,i,par->local_stb[i].name);
        int32_t s;
		gwf_path_t path = par->ptb[i];
		s = gwf_ed(km, g, par->local_stb[i].l, par->local_stb[i].s, 0, -1, max_lag, traceback, &path, my_rank, par->my_rank);
		if (traceback) {
			int32_t j, last_len = -1, len = 0;
			printf("%s\t%ld\t0\t%ld\t+\t", par->local_stb[i].name, par->local_stb[i].l, par->local_stb[i].l);
			for (j = 0; j < path.nv; ++j) {
				uint32_t v = g->src[path.v[j]];
				printf("%c%s", "><"[v&1], gfa->seg[v>>1].name);
				last_len = gfa->seg[v>>1].len;
				len += last_len;
			}
			printf("\t%d\t0\t%d\t%d\n", len, len - (last_len - path.end_off) + 1, path.s);
		} else {
            printf("%s\t%d,thread %ld,proc%d\n", par->local_stb[i].name, s, my_rank, par->my_rank);
        }

		// printf("%s\n",stb[i].s);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    thread_elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("thread_time: %f s from thread %ld, proc%d\n", thread_elapsed, my_rank, par->my_rank);
    printf("total_s_expand_time: %f s from thread %ld, proc%d\n", total_s_expand_time, rank, par->my_rank);
    printf("total_p_expand_time: %f s from thread %ld, proc%d\n", total_p_expand_time, rank, par->my_rank);
    printf("total_s_extend_time: %f s from thread %ld, proc%d\n", total_s_extend_time, rank, par->my_rank);
    printf("total_p_extend_time: %f s from thread %ld, proc%d\n", total_p_extend_time, rank, par->my_rank);
    printf("total_s_lcp_time: %f s from thread %ld, proc%d\n", total_s_lcp_time, rank, par->my_rank);
    printf("total_p_lcp_time: %f s from thread %ld, proc%d\n", total_p_lcp_time, rank, par->my_rank);
	// printf("Hello from thread number %ld(on %ld)for the MPI process number %d (on %d)\n",my_rank, thread_count, par->my_rank, par->comm_sz);
    return NULL;
}
void* Thread_worker(void* rank) {
    struct timespec start, end;
    double thread_elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while(1){

//        pthread_mutex_lock(&lock); // 加锁，开始临界区域
//
////         当任务队列为空时等待任务到来
//        while (task_index[par->my_rank] > par->local_n +1) {
//            pthread_cond_wait(&cond, &lock); // 等待条件变量，释放锁并阻塞线程
//        }
        // 取出一个任务并更新任务索引
        task_index[par->my_rank]++;
        pthread_mutex_unlock(&lock); // 解锁，结束临界区域

        // 执行任务
        long my_rank = (long) rank;
//        printf("now task_index is %d of par->local_n: %d of thread %ld , proc %d\n",task_index[par->my_rank], par->local_n, my_rank,par->my_rank);
        int32_t s;
        gwf_path_t path = par->ptb[task_index[par->my_rank]-1];
        s = gwf_ed(km, g, par->local_stb[task_index[par->my_rank]-1].l, par->local_stb[task_index[par->my_rank]-1].s, 0, -1, max_lag, traceback, &path, my_rank, par->my_rank);
        if (traceback) {
            int32_t j, last_len = -1, len = 0;
            printf("%s\t%ld\t0\t%ld\t+\t", par->local_stb[task_index[par->my_rank]-1].name, par->local_stb[task_index[par->my_rank]-1].l, par->local_stb[task_index[par->my_rank]-1].l);
            for (j = 0; j < path.nv; ++j) {
                uint32_t v = g->src[path.v[j]];
                printf("%c%s", "><"[v&1], gfa->seg[v>>1].name);
                last_len = gfa->seg[v>>1].len;
                len += last_len;
            }
            printf("\t%d\t0\t%d\t%d\n", len, len - (last_len - path.end_off) + 1, path.s);
        } else {
            printf("%s\t%d,thread %ld,proc%d\n", par->local_stb[task_index[par->my_rank]-1].name, s, my_rank, par->my_rank);
        }


        // 如果任务是最后一个任务，结束工作线程
        if (task_index[par->my_rank] == par->local_n) {
//            printf("pcc_debug\n");
            break;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    thread_elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("thread_time: %f s from thread %ld, proc%d\n", thread_elapsed, rank, par->my_rank);
    pthread_exit(NULL); // 退出线程
}