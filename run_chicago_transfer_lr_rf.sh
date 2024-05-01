for i in 1 2 3 4 5 6 7 8 9 10
do
    nohup python -W ignore src_gat_transfer/mains/lr_rf_transfer.py -c src_gat_transfer/configs/venue_sex_chicago_fold${i}.json -t 20 > results/chicago_transfer_lr_rf_reduced_1.0_0.3/chicago.fold${i} 2>&1&
done
