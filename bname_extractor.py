import os
import numpy as np

sample_size = 30
book_dir = "./books"
outfname = "book_sample.txt"

if __name__ == "__main__":
    all_books = os.listdir(book_dir)
    all_books = np.array(all_books)
    num_books = len(all_books)
    book_sample = np.random.choice(num_books, sample_size)
    sample_books = all_books[book_sample]
    outfile = open(outfname, 'w')
    for idx in range(sample_books.shape[0]):
        sample_book = sample_books[idx]
        outfile.write(sample_book + '\n')
    outfile.close()
