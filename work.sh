time build/bin/task2 -d ../data/binary/train_labels.txt -m model.txt --train --$1;
echo "Training done";
time build/bin/task2 -d ../data/binary/test_labels.txt -m model.txt -l predictions.txt --predict --$1;
echo "Prediction done";
python compare.py ../data/binary/test_labels.txt predictions.txt


#time build/bin/task2 -d ../data/multiclass/train_labels.txt -m model_multiclass.txt --train;
#echo "Multiclass training done";
#time build/bin/task2 -d ../data/multiclass/test_labels.txt -m model_multiclass.txt -l predictions.txt --predict;
#echo "Multiclass prediction done";
#python compare.py ../data/multiclass/test_labels.txt predictions.txt;
