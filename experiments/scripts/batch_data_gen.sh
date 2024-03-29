#!/bin/bash

##########################################################
# Run this script from the experiments/scripts subdirectory
# Activate an appropriate python env before running
# Comment or uncomment below as desired
##########################################################

##############################
# generate some sample files for debugging.
##############################

d=5 # dim 5 only
sk=0 # no skew
for ((n=256; n<1025; n=n*4))
do
    for ((fr=0; fr<2; fr=fr+1)); do
        for ((seed=0; seed<3; seed=seed+1)); do
            # echo ${d} ${n} ${sk} ${fr} ${seed}
            python data_gen.py --d ${d} --n ${n} --skewness ${sk} --frequency ${fr}  --seed ${seed}
        done
    done
done


##############################
# vary over n and skew
##############################

# d=5
# fr=0
# for ((n=256; n<16385; n=n*4))
# do
#     for ((sk=0; sk<11; sk=sk+10)); do
#         for ((seed=0; seed<5; seed=seed+1)); do
#             # echo ${d} ${n} ${sk} ${fr} ${seed}
#             python generators/data_gen.py --d ${d} --n ${n} --skewness ${sk} --frequency ${fr} --seed ${seed}
#         done
#     done
# done

##############################
# vary over n and frequency
##############################

# d=5
# sk=0
# for ((n=256; n<16385; n=n*4))
# do
#     for ((fr=0; fr<2; fr=fr+1)); do
#         for ((seed=0; seed<5; seed=seed+1)); do
#             # echo ${d} ${n} ${sk} ${fr} ${seed}
#             python generators/data_gen.py --d ${d} --n ${n} --skewness ${sk} --frequency ${fr}  --seed ${seed}
#         done
#     done
# done

##############################
# vary over d and frequency
##############################

# n=8192
# sk=0
# for ((d=2; d<21; d=d+2))
# do
#     for ((fr=0; fr<2; fr=fr+1)); do
#         for ((seed=0; seed<5; seed=seed+1)); do
#             # echo ${d} ${n} ${sk} ${fr} ${seed}
#             python generators/data_gen.py --d ${d} --n ${n} --skewness ${sk} --frequency ${fr} --seed ${seed}
#         done
#     done
# done

##############################
# Make 10 seeds but leave everything else fixed - for debugging 
##############################

# d=5
# sk=0
# n=1024
# fr=0
# for ((seed=0; seed<10; seed=seed+1)); do
#     # echo ${d} ${n} ${sk} $(bc<<<"$step * $fr") ${seed}
#     python generators/data_gen.py --d ${d} --n ${n} --skewness ${sk} --frequency ${fr}  --seed ${seed}
# done

##############################
# vary over n and dim and seed
##############################

# fr=0
# sk=0
# for ((n=256; n<16385; n=n*4)); do
#     for ((d=2; d<11; d=d+2)); do
#         for ((seed=0; seed<10; seed=seed+1)); do
#             echo ${d} ${n} ${sk} ${fr} ${seed}
#             # python generators/data_gen.py --d ${d} --n ${n} --skewness ${sk} --frequency ${fr} --seed ${seed}
#         done
#     done
# done

##############################
# vary over d and n for extrapp;atopm queries
##############################

# fr=0
# sk=0
# seed=0
# for ((d=11; d<12; d=d+3)); do
#     for ((n=256; n<16385; n=n*4)); do
#         # echo ${d} ${n} ${sk} ${fr} ${seed} lhc
#         python generators/data_gen.py --d ${d} --n ${n} --skewness ${sk} --frequency ${fr} --seed ${seed} --spacing sob
#         # python generators/data_gen.py --d ${d} --n ${n} --skewness ${sk} --frequency ${fr} --seed ${seed} --spacing lhc
#         # python generators/data_gen.py --d ${d} --n ${n} --skewness ${sk} --frequency ${fr} --seed ${seed} --spacing uni
#     done
# done
