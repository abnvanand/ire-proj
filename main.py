import pandas as pd
from src.constants import FIELD_SEP_ESCAPED

print(FIELD_SEP_ESCAPED)

df = pd.read_csv(filepath_or_buffer='processedData/articles-training-byarticle.txt',
                 sep=FIELD_SEP_ESCAPED,
                 names=['article_id', 'title', 'articleContent', 'hyperpartisan']
                 )
print(df.head())
