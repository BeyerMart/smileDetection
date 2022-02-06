import os

def generate_negative_bg_file():
    with open('dataset/neg.txt', 'w') as file:
        for filename in os.listdir('dataset/negatives'):
            file.write('negatives/' + filename + '\n')

generate_negative_bg_file()