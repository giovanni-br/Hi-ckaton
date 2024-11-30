import pandas as pd

def process_chunk(chunk):
    # Apply your preprocessing steps on the chunk
    chunk, enc, var = preprocess_yann.preprocess_and_clean_yann(chunk)
    chunk, enc, var = preprocess_oliv.clean_olivier(chunk)
    chunk, enc, var = preprocess_gio.preprocess_and_clean_gio(chunk)
    chunk, enc, var = preprocess_alejandro.encodeAlejandroVariables(chunk)
    chunk, enc, var = preprocess_yassine.preprocess_and_clean_Yassine(chunk)
    return chunk

chunksize = 10000  # Define an appropriate chunk size
processed_data = []

for chunk in pd.read_csv('X_train_Hi5.csv', chunksize=chunksize):
    processed_chunk = process_chunk(chunk)
    processed_data.append(processed_chunk)

# Combine processed chunks into a single DataFrame
final_data = pd.concat(processed_data, ignore_index=True)
