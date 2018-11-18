import re       #regular expression       
def read_book(title_path):
    '''read a book and return it as a string'''
    with open(title_path, "r") as current_file: 
        text = current_file.read()
        text = text.replace('\n','').replace('\r','')
    return text


from collections import Counter
def count_fast(text):
    '''Count the unique words into  a dictionary with
    values as the frequency of the word[key]'''
    p = re.compile("[^A-z ]")
    text = p.sub("",text).lower()
    word_counts = Counter(text.split(" "))#counter creates dictionary key with
    return word_counts # words and counts how many times the keys appeared in
                       #the text, making it the values of the dictionary                   


def word_stats(word_counts):
    '''return number of unique words and word frequency'''
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)


import os   
book_dir = "./Books"

#uses pandas, initial for panel datas,
#used to refer to multi-dimensional structured data sets.
import pandas as pd
'''
This creates a table with two rows, row one at loc[1] etc
table = ps.DataFrame(columns = ("name","age"))
table.loc[1] = "James",22
table.loc[2] = "Jess",15
'''

stats = pd.DataFrame(columns = ("language", "author", "title", "length","unique"))
title_num = 1
for language in os.listdir(book_dir)[1:]:
    for author in os.listdir(book_dir + "/" + language)[1:]:
        for title in os.listdir(book_dir + "/" + language + "/" + author):
            inputfile = book_dir + "/" + language + "/" + author + "/" + title
            text = read_book(inputfile)
            (num_unique, counts) = word_stats(count_fast(text))
            stats.loc[title_num] = language, author.capitalize(),title.replace(".txt",""), sum(counts), num_unique
            title_num += 1


#plot the data into  a graph
import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))
subset = stats[stats.language == "English"]
plt.loglog(subset.length,subset.unique, "o", label = "English", color = "crimson")
subset = stats[stats.language == "French"]
plt.loglog(subset.length,subset.unique, "o", label = "French", color = "forestgreen")
subset = stats[stats.language == "German"]
plt.loglog(subset.length,subset.unique, "o", label = "German", color = "orange")
subset = stats[stats.language == "Portuguese"]
plt.loglog(subset.length,subset.unique, "o", label = "Portuguese", color = "blueviolet")
plt.legend()
plt.xlabel("Book length")
plt.ylabel("Number of unique words")
plt.savefig("lang_plot.pdf")
            
