build/bin/task2 -d ../data/binary/train_labels.txt -m model.txt --train;
echo "Training done";
build/bin/task2 -d ../data/binary/test_labels.txt -m model.txt -l predictions.txt --predict;
echo "Prediction done";
python compare.py ../data/binary/test_labels.txt predictions.txt


build/bin/task2 -d ../data/multiclass/train_labels.txt -m model_multiclass.txt --train;
echo "Multiclass training done";
build/bin/task2 -d ../data/multiclass/test_labels.txt -m model_multiclass.txt -l predictions.txt --predict;
echo "Multiclass prediction done";
python compare.py ../data/multiclass/test_labels.txt predictions.txt;
