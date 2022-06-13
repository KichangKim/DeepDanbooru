# DeepDanbooru
[![Python](https://img.shields.io/badge/python-3.6-green)](https://www.python.org/doc/versions/)
[![GitHub](https://img.shields.io/github/license/KichangKim/DeepDanbooru)](https://opensource.org/licenses/MIT)
[![Web](https://img.shields.io/badge/web%20demo-20200915-brightgreen)](http://kanotype.iptime.org:8003/deepdanbooru/)

**DeepDanbooru** is anime-style girl image tag estimation system. You can estimate your images on my live demo site, [DeepDanbooru Web](http://dev.kanotype.net:8003/deepdanbooru/).

## Requirements
DeepDanbooru is written by Python 3.7. Following packages are need to be installed.
- tensorflow>=2.7.0
- tensorflow-io>=2.22.0
- Click>=7.0
- numpy>=1.16.2
- requests>=2.22.0
- scikit-image>=0.15.0
- six>=1.13.0

Or just use `requirements.txt`.
```
> pip install -r requirements.txt
```

alternatively you can install it with pip. Note that by default, tensorflow is not included.

To install it with tensorflow, add `tensorflow` extra package.

```
> # default installation
> pip install .
> # with tensorflow package
> pip install .[tensorflow]
```


## Usage
1. Prepare dataset. If you don't have, you can use [DanbooruDownloader](https://github.com/KichangKim/DanbooruDownloader) for download the dataset of [Danbooru](https://danbooru.donmai.us/). If you want to make your own dataset, see [Dataset Structure](#dataset-structure) section.
2. Create training project folder.
```
> deepdanbooru create-project [your_project_folder]
```
3. Prepare tag list. If you want to use latest tags, use following command. It downloads tag from Danbooru server.
```
> deepdanbooru download-tags [your_project_folder]
```
4. (Option) Filtering dataset. If you want to train with optional tags (rating and score), you should convert it as system tags.
```
> deepdanbooru make-training-database [your_dataset_sqlite_path] [your_filtered_sqlite_path]
```
5. Modify `project.json` in the project folder. You should change `database_path` setting to your actual sqlite file path.
6. Start training.
```
> deepdanbooru train-project [your_project_folder]
```
7. Enjoy it.
```
> deepdanbooru evaluate [image_file_path or folder]... --project-path [your_project_folder] --allow-folder
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
