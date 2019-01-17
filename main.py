import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

md_path = "modules/universal-sentence-encoder" 
md = hub.Module(md_path)

placeholder = tf.placeholder(tf.string, shape=(None))
encoding = md(placeholder)

std = 0.5

corpus = [
    "I like my phone",
    "Will it snow tomorrow?",
    "An apple a day, keeps the doctors away",
    "How old are you?",
]

data = [
    "My phone is not good.",
    "Your cellphone looks great.",
    "Recently a lot of hurricanes have hit the US",
    "Global warming is real",
    "Eating strawberries is healthy",
    "Is paleo better than keto?",
    "what is your age?",
]

def draw(inner_product):
    sns.set(font_scale=0.6)
    g = sns.heatmap(
        inner_product,
        xticklabels=data,
        yticklabels=corpus,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu")
    g.set_xticklabels(data, rotation=90)
    g.set_title("Semantic Textual Similarity")
    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

    corpus_analysis = session.run(encoding, feed_dict={placeholder: corpus})
    data_analysis = session.run(encoding, feed_dict={placeholder: data})

    inner_product = np.inner(corpus_analysis, data_analysis)
    similarity = inner_product.copy()
    similarity[inner_product < std] = 0
    similarity[inner_product > std] = 1

    result = []
    for y in range(len(corpus)):
        for x in range(len(data)):
            if similarity[y][x] == 1:
                result.append("%s # %s # %.2f" % (corpus[y], data[x], inner_product[y][x]))

    for i in result:
        print(i)
    
    draw(inner_product)