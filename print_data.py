import textwrap

from sklearn.datasets import fetch_20newsgroups

def main(subset):
    bunch = fetch_20newsgroups(
        subset=subset, remove=('headers', 'footers', 'quotes'))
    
    labels = bunch['target_names']
    for label_idx, txt in zip(bunch.target, bunch.data):
        lstr = 'Label: {}'.format(labels[label_idx])
        print lstr
        print '=' * len(lstr)
        print textwrap.fill(txt.encode(u'utf-8'), replace_whitespace=False)
        print


if __name__ == u'__main__':
    subset = u'train'
    main(subset)
