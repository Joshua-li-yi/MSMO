# -*- coding:utf-8 -*-
# @Time： 2020-07-02 15:55
# @Author: Joshua_yi
# @FileName: make_datafiles.py
# @Software: PyCharm
# @Project: MSMO
# @Description: product the used data for train

import ujson
import os
import config
import subprocess
import hashlib
import collections

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

root_dir = config.root_dir + r'data/data_preview/'
all_urls = root_dir + r"url/url_list"

tokenized_articles_dir = root_dir + r"tokenized/"
finished_files_dir = root_dir +r"msmo_data/"
train_data_dir = finished_files_dir + r'train/'
valid_data_dir = finished_files_dir + r'valid/'
test_data_dir = finished_files_dir + r'test/'
articles_dir = root_dir + r'article/'
imgs_dir = root_dir + r'img/'

chunks_dir = os.path.join(finished_files_dir, r"/chunked")

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_articles = 200

VOCAB_SIZE = config.vocab_size
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def check_num_articles(stories_dir, num_expected):
    """
    判断文件的数量是否符合要求
    :param stories_dir: str 文件的路径
    :param num_expected: int 文件的数量
    :return:
    """
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


def read_text_file(text_file):
    """
    读取txt文件
    :param text_file: str 文件的路径
    :return: list 所读取的txt
    """
    lines = []
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def read_url_file(url_file):
    lines = []
    with open(url_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    # 截取所需要的url 数量
    # 用来做测试时使用
    # 如果是正式运行，请注释调改行
    lines = lines[:num_expected_articles]
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@summary" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def get_art_abs(story_file):
    """
    读取token后的文件并返回article 和 abstract
    :param story_file: str 文件的路径
    :return: article： str
            abstract: str
    """
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@summary"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return article, abstract


def tokenize_articles(stories_dir, tokenized_stories_dir, url_file=None, output_file=None, delete_map=True, make_vocab=True):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    print("Making list of files to tokenize...")
    with open(root_dir+"mapping.txt", "w") as f:
        for id, s in enumerate(stories):
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', root_dir+'mapping.txt']

    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)

    print("Stanford CoreNLP Tokenizer has finished.")

    if delete_map: os.remove(root_dir+"mapping.txt")

    article_imgs = {}
    flag = 0
    imgs_list = os.listdir(imgs_dir)
    for story in stories:
        imgs = imgs_list[flag: flag + config.maxinum_imgs]
        # for img in os.listdir(imgs_dir):
        #     if story[:-4] == img[:len(story[:-4])]:
        #         # print(img)
        #         img = imgs_dir + img
        #         imgs.append(img)
        flag += config.maxinum_imgs
        article_imgs[story] = imgs

    # 如果是要统计词频
    if make_vocab:
        vocab_counter = collections.Counter()

    tokenized_stories = os.listdir(tokenized_articles_dir)
    train_list = []
    for story_path in tokenized_stories:
        train_dict = {}
        # Get the strings to write to .bin file
        train_dict['imgs'] = article_imgs[story_path]
        story_path = articles_dir + story_path
        article, abstract = get_art_abs(story_path)
        train_dict['article'] = article
        train_dict['abstract'] = abstract

        train_list.append(train_dict)

        # Write the vocab to file, if applicable
        if make_vocab:
            art_tokens = article.split(' ')
            abs_tokens = abstract.split(' ')
            abs_tokens = [t for t in abs_tokens if
                          t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if t != ""]  # remove empty
            vocab_counter.update(tokens)

    print(len(train_list))

    # Writing JSON data
    with open(train_data_dir+'train_ATL_HAN.json', 'w') as f:
        ujson.dump(train_list[:150], f, indent=4)

    with open(valid_data_dir + 'valid_ATL_HAN.json', 'w') as f:
        ujson.dump(train_list[150:175], f, indent=4)

    with open(test_data_dir+'test_ATL_HAN.json', 'w') as f:
        ujson.dump(train_list[175:], f, indent=4)

    print("Successfully save data ")

    # write vocab to file
    if make_vocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab.txt"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")
    pass


if __name__ == '__main__':

    check_num_articles(articles_dir, num_expected_articles)
    # Create some new directories
    if not os.path.exists(tokenized_articles_dir): os.makedirs(tokenized_articles_dir)
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
    if not os.path.exists(train_data_dir): os.makedirs(train_data_dir)
    if not os.path.exists(valid_data_dir): os.makedirs(valid_data_dir)
    if not os.path.exists(test_data_dir): os.makedirs(test_data_dir)
    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories, and the final data
    tokenize_articles(articles_dir, tokenized_articles_dir, url_file=all_urls, delete_map=False)