root_folder=$1
#dcase not working now so test only on speech_commands
task=speech_commands
module=hearbaseline
model_download_path=https://github.com/neuralaudio/hear-baseline/raw/main/saved_models/naive_baseline.pt
model_name=naive_baseline.pt

luigi_dir=$root_folder/test_luigi
tasks_dir=$root_folder/test_tasks
embeddings_dir=$root_folder/test_embeddings
model_path=$root_folder/$model_name

echo "Luigi Dir $luigi_dir"
echo "Task Dir $tasks_dir"
echo "Embeddings Dir $embeddings_dir"

#COMPUTE EMBEDDINGS
pip install -e .
pip install module
#STEP0
#Run the luigi pipeline task to generate consumable data
python -m heareval.tasks.runner $task \
    --small \
    --luigi-dir $luigi_dir \
    --tasks-dir $tasks_dir \
#STEP1
#Download the model
wget -N $model_download_path -P $root_folder
#STEP2
# Produce embeddings with the model module for the generated data from step1
python3 -m heareval.embeddings.runner $module \
    --model $model_path \
    --tasks-dir $tasks_dir \
    --embeddings-dir $embeddings_dir

#DOWNSTREAM EVALUATION
#STEP1
#Produce prediction for the test set with the embeddings from step2
python3 -m heareval.predictions.runner $module \
    --model $model_path \
    --embeddings-dir $embeddings_dir

#STEP2
#Evaluate the results produced by the step4
python3 -m heareval.evaluation.runner \
    --embeddings-dir $embeddings_dir
