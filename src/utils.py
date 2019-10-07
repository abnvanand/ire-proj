import re

nbsp = re.compile('&#160;')
spaces = re.compile("\\s+")


def clean_text(text):
    text = text.replace("&amp;", "&")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace(" _", " ")
    text = text.replace("–", "-")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("’", "'")

    text = nbsp.sub(' ', text)
    text = spaces.sub(' ', text)
    return text
