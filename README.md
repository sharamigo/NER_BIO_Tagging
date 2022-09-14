# NER_BIO_Tagging
Prototype for a Named Entity Recognition for textual data in a machine learning project (Quenfo)

## Folder structure:

```
project
│   README.md
│   ner_bio_tagger.py
|   ner_model.py
|   sentence.py
|   utils.py
|   model.h5
|   tag_to_index.pickle
|   word_to_index.pickle 
│
└───data
│   │   001_2016_08_JobAd_export.csv
│   │   002_2016_08_JobAd_export.csv
│   │   ...
│   └───text-files
│       │   001_2016_08_116dc8a4be404996b29cb87ea4ee9116.txt 
│       │   002_2016_08_15e23d3b22d844f7a67cc4ed00208439.txt
│       │   ...
|   └───xmi
|       |  001_2018_08_JobAd.xmi
|       |  005_2016_08_JobAd.xmi
|       |  ...
│   
└───output
    │   file021.txt
    │   file022.txt
```

**data-Folder** contains all exported CSV-files from annotation-process.
