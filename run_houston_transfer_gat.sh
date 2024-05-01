for i in 1 2 3 4 5 6 7 8 9 10
do
    nohup python -W ignore src_gat_transfer/mains/gat_main_for_hiv.py -c src_gat_transfer/configs/venue_sex_houston_fold${i}.json -t 20 > results/houston_transfer_gat_reduced_sex_1.0_0.3/houston.fold${i} 2>&1&
done
