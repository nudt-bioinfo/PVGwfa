# PVGwfa: Multi-Level Parallel Sequence-to-Graph Alignment Tool

## Overview  
A pangenome graph represents the genomes of multiple individuals, offering a comprehensive reference and overcoming allele bias from linear reference genomes. Sequence-to-graph alignment, crucial for pangenome tasks, aligns sequences to a graph to find the best matches. However, existing algorithms struggle with large-scale sequences.

PVGwfa is a multi-level parallel sequence-to-graph alignment algorithm designed to address these challenges:  
- Employs **MPI** and **Pthread** for multi-process and multi-thread parallelization.  
- Introduces a **hybrid load balancing strategy** for better performance.  
- Leverages **SIMD vectorization** to accelerate sequence alignment.  

### Highlights:  
- Significant speedups: From nearly an hour to just a few minutes for large datasets.  
- Scalable performance: Achieved **1.98x to 100.44x speedups** as the number of processes increased from 2 to 128.  
- Maintains consistent alignment results across experiments.  

---

## Getting Started  

### Prerequisites  
Before compiling PVGwfa, ensure your system has:  
- **MPI** installed (e.g., OpenMPI, MPICH).  
- **Pthread** support.  

### Installation  
Clone the repository and compile using `make`:  
```bash  
git clone https://github.com/nudt-bioinfo/PVGwfa.git  
cd PVGwfa && make
```

---
### Usage
Run PVGwfa using the following command:
```bash
mpirun -n <number_of_processes> ./PVGwfa <compressed_gfa_file> <compressed_fasta_file> <number_of_threads>  
mpirun -n 4 ./PVGwfa data/sample.gfa.gz data/sample.fa.gz 8  
```
---
### License
PVGwfa is an enhanced version of the Gwfa tool, which is distributed under the MIT License. To align with this, PVGwfa is also released under the MIT License.
