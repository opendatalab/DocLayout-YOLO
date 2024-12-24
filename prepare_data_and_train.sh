# Connect to HuggingFace hub if not already connected
if [ ! -f ~/.huggingface/token ]; then
    huggingface-cli login
fi

# Prepare data for training
python prepare_data_for_training.py

# Train the model
python training_wrapper.py --push