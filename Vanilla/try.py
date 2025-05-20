import pandas as pd

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv("samples/vanilla_bs:64_bs:3_b:False_ct:GRU_dnl:3_d:0.3_ed:32_enl:3_e:20_hu:256_lr:0.003_.txt", sep="\t", header=None)
    correct = []
    incorrect = []
    
    np_df = df.to_numpy()
    for i in range(len(np_df)):
        if np_df[i][1] == np_df[i][2] and len(correct) <= 10:
            correct.append(np_df[i])
        if np_df[i][1] != np_df[i][2] and len(incorrect) <= 10:
            incorrect.append(np_df[i])
        if len(correct) > 10 and len(incorrect) > 10:
            break
    print("Correct Predictions:")
    for i in range(len(correct)):
        print(correct[i][1], correct[i][2])
    print("Incorrect Predictions:")
    for i in range(len(incorrect)):
        print(incorrect[i][1], incorrect[i][2])
        
    with open('correct.txt', 'w') as f:
        for i in range(len(correct)):
            f.write(f"{correct[i][0]}\t{correct[i][1]}\t{correct[i][2]}\n")
    with open('incorrect.txt', 'w') as f:
        for i in range(len(incorrect)):
            f.write(f"{incorrect[i][0]}\t{incorrect[i][1]}\t{incorrect[i][2]}\n")
    print("Correct and incorrect predictions saved to correct.txt and incorrect.txt respectively.")
    