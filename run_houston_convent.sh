for i in 1 2 3 4 5 6 7 8 9 10
do
    nohup python -W ignore src_convent/conventional_ml_hiv.py -c src_convent/configs/venue_sex_houston_fold${i}.json  > results/houston_convent_0.75/reduce_graph/houston.fold${i} 2>&1&
done
