import xml.sax
import csv
from src.utils import clean_text


class ArticleHandler(xml.sax.ContentHandler):
    def __init__(self, outfile, ground_truth):
        self.article_id = ''
        self.title = ''
        self.published_at = ''
        self.articleContent = []
        self.outfilefp = open(outfile, "w")
        self.csvwriter = csv.writer(self.outfilefp)
        self.ground_truth = ground_truth

    def startElement(self, tag, attributes):
        if tag == 'article':
            self.article_id = attributes.getValue('id')
            # FIXME: some articles don't have published_at so not writing to cleaned file
            # self.published_at = attributes.getValue('published-at')
            self.title = attributes.getValue('title')
            self.articleContent.clear()

    def endElement(self, tag):
        if tag == 'article':
            content = " ".join(self.articleContent)
            content = clean_text(content)
            print(self.article_id)
            row = [self.article_id,
                   self.title,
                   content,
                   self.ground_truth[self.article_id]]
            self.csvwriter.writerow(row)

    def characters(self, content):
        self.articleContent.append(content)

    def endDocument(self):
        self.outfilefp.close()


groundTruth = {}


class GroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.article_id = ''
        self.hyperpartisan = ''
        self.labeled_by = ''
        self.url = ''
        self.bias = ''

    def startElement(self, tag, attributes):
        if tag == "article":
            self.article_id = attributes.getValue('id')
            self.hyperpartisan = attributes.getValue('hyperpartisan')
            self.labeled_by = attributes.getValue('labeled-by')
            self.url = attributes.getValue('url')
            if 'bias' in attributes:
                self.bias = attributes.getValue('bias')
            groundTruth[self.article_id] = self.hyperpartisan


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # STOPSHIP remove defaults
    parser.add_argument('--groundTruthsFilePath', '-g', default="data/ground-truth-training-byarticle-20181122.xml")
    parser.add_argument('--articlesFilePath', '-a', default="data/articles-training-byarticle-20181122.xml")
    parser.add_argument('--outputFilePath', '-o', default="processedData/articles-training-byarticle.csv")
    args = parser.parse_args()

    parser = xml.sax.make_parser()

    # turn off namespaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    handlerGroundTruth = GroundTruthHandler()
    parser.setContentHandler(handlerGroundTruth)
    parser.parse(args.groundTruthsFilePath)

    handler = ArticleHandler(args.outputFilePath, groundTruth)
    parser.setContentHandler(handler)

    parser.parse(args.articlesFilePath)
