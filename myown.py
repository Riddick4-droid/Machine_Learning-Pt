from matplotlib import pyplot as plt
from matplotlib import style
from itertools import islice
from collections import defaultdict
style.use('fivethirtyeight')

import re

def plot_Zipfs_Law(filenames):
    '''Plots Zipf's Law and handles the zero-frequency problem across multiple files.'''

    all_frequencies = []
    combined_vocab = set()

    for filename in filenames:
        frequency = {}

        with open(filename, 'r', encoding="UTF8") as content:
            text_string = content.read()

            words = re.findall(r'\b[A-Za-z][a-z]{2,9}\b', text_string)

            for word in words:
                frequency[word] = frequency.get(word, 0) + 1

            all_frequencies.append(frequency)
            combined_vocab.update(frequency.keys())

    # Compare and find missing words across datasets
    print("Zero Frequency Analysis:")

    for i, current_freq in enumerate(all_frequencies):
        other_vocab = set().union(*[f for j, f in enumerate(all_frequencies) if j != i])

        missing_words = other_vocab - set(current_freq.keys())
        print(f"In {filenames[i]}, {len(missing_words)} words are missing from other datasets.")

        # Frequency analysis
        count_1 = sum(1 for freq in current_freq.values() if freq == 1)
        count_2 = sum(1 for freq in current_freq.values() if freq == 2)
        count_2m = sum(1 for freq in current_freq.values() if freq > 2)
        total_words = sum(current_freq.values())

        print(f"Total words in {filenames[i]}: {total_words}")
        print(f"Words with frequency 1: {count_1} ({(count_1/total_words)*100:.2f}%)")
        print(f"Words with frequency 2: {count_2} ({(count_2/total_words)*100:.2f}%)")
        print(f"Words with frequency >2: {count_2m} ({(count_2m/total_words)*100:.2f}%)")
        print("-" * 50)

        # Plotting
        ranked_freqs = sorted(current_freq.values(), reverse=True)
        plt.plot(range(1, len(ranked_freqs) + 1), ranked_freqs, label=filenames[i])

    plt.title("Zipf's Law for Different Texts")
    plt.legend()
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.xscale('log')
    plt.yscale('log')

    plt.show()

# Example usage - replace with your actual filenames
plot_Zipfs_Law(["Bible.txt", "mytext1.txt", "declaration.txt"])

