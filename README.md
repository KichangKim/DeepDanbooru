# DeepDanbooru
[![Python](https://img.shields.io/badge/python-3.6-green)](https://www.python.org/doc/versions/)
[![GitHub](https://img.shields.io/github/license/KichangKim/DeepDanbooru)](https://opensource.org/licenses/MIT)
[![Web](https://img.shields.io/badge/web%20demo-20191108-brightgreen)](http://kanotype.iptime.org:8003/deepdanbooru/)

**DeepDanbooru** is anime-style girl image tag estimation system. You can estimate your images on my live demo site, [DeepDanbooru Web](http://kanotype.iptime.org:8003/deepdanbooru/).

## Requirements
DeepDanbooru is written by Python 3.6. Following packages are need to be installed.
- tensorflow==2.1.0rc1
- Click==7.0
- scikit-image==0.15.0
- numpy==1.16.2+mkl

Or just use `requirements.txt`.
```
> pip install -r requirements.txt
```

## Usage
1. Prepare dataset. If you don't have, you can use [DanbooruDownloader](https://github.com/KichangKim/DanbooruDownloader) for download the dataset of [Danbooru](https://danbooru.donmai.us/). If you want to make your own dataset, see [Dataset Structure](#dataset-structure) section.
2. Create training project folder.
```
> python program.py create-project [your_project_folder]
```
3. Prepare tag list. If you want to use latest tags, use following command. It downloads tag from Danbooru server.
```
> python program.py download-tags [your_project_folder]
```
4. (Option) Filtering dataset. If you want to train with optional tags (rating and score), you should convert it as system tags.
```
> python program.py make-training-database [your_dataset_sqlite_path] [your_filtered_sqlite_path]
```
5. Modify `project.json` in the project folder. You should change `database_path` setting to your actual sqlite file path.
6. Start training.
```
> python program.py train-project [your_project_folder]
```
7. Enjoy it.
```
> python program.py evaluate-project [your_project_folder] [image_file_path]
```

## Dataset Structure
DeepDanbooru uses following folder structure for input dataset. SQLite file can be any name, but must be located in same folder to `images` folder.
```
MyDataset/
├── images/
│   ├── 00/
│   │   ├── 00000000000000000000000000000000.jpg
│   │   ├── ...
│   ├── 01/
│   │   ├── ...
│   └── ff/
│       ├── ...
└── my-dataset.sqlite
```
The core is SQLite database file. That file must be contains following table structure.
```
posts
├── id (INTEGER)
├── md5 (TEXT)
├── file_ext (TEXT)
├── tag_string (TEXT)
└── tag_count_general (INTEGER)
```
The filename of image must be `[md5].[file_ext]`. If you use your own images, `md5` don't have to be actual MD5 hash value.

`tag_string` is space splitted tag list, like `1girl ahoge long_hair`.

`tag_count_general` is used for the project setting, `minimum_tag_count`. Images which has equal or larger value of `tag_count_general` are used for training.

## Project Structure
**Project** is minimal unit for training on DeepDanbooru. You can modify various parameters for training.
```
MyProject/
├── project.json
└── tags.txt
```
`tags.txt` contains all tags for estimating. You can make your own list or download latest tags from Danbooru server. It is simple newline-separated file like this:
```
1girl
ahoge
...
```
