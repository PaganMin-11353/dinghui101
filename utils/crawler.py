import os
import json
import logging
import time
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import requests
from dotenv import load_dotenv
import shelve
import pandas as pd
from tqdm import tqdm

load_dotenv()

DATASET_SIZE = '100k'
BASE_DIR = os.path.join('data', 'augmented', f'ml-{DATASET_SIZE}')
POSTER_DIR = os.path.join(BASE_DIR, 'posters')
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, 'fetched_movies.csv')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
REQUEST_DELAY = 0.25  # 4 requests per second

DOWNLOAD_POSTER = False
POSTER_SIZE = 'w92'  # Available sizes: w92, w154, w185, w342, w500, w780, original

MOVIELENS_CSV_PATH = f'data/ml-{DATASET_SIZE}/u.item'


class MovieDataCrawler:
    def __init__(self):
        self.setup_directories()
        self.setup_logging()
        
        self.tmdb_api_key = os.getenv('TMDB_API_KEY')
        if not self.tmdb_api_key:
            self.logger.critical("TMDB_API_KEY not found in environment variables!")
            raise ValueError("TMDB_API_KEY environment variable not set. Please add it to your .env file.")
        
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {
            'Authorization': f'Bearer {self.tmdb_api_key}',
            'accept': 'application/json'
        }
        
        self.stats = {
            'total_movies': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'cache_hits': 0,
            'parsing_errors': 0,
            'api_errors': 0,
            'poster_downloads': 0,
            'poster_failures': 0
        }

    def setup_directories(self):
        """Create all necessary directories"""
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(POSTER_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

    def setup_logging(self):
        """Setup enhanced logging with both file and console output"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(LOG_DIR, f'crawler_{timestamp}.log')
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s\n'
            'Additional Info: %(pathname)s:%(lineno)d\n'
            'Exception Info: %(exc_info)s\n'
        )
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        self.logger = logging.getLogger("crawler")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make an API request with error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}", exc_info=True)
            return None

    def parse_movie_title(self, title: str) -> Tuple[str, Optional[str]]:
        """Parse movie title and extract year if present"""
        self.logger.debug(f"Parsing title: {title}")
        
        if not title:
            self.logger.warning("Empty title provided")
            self.stats['parsing_errors'] += 1
            return "", None

        try:
            if '(' in title and ')' in title:
                year_part = title[title.rindex('(')+1:title.rindex(')')]
                if year_part.isdigit() and len(year_part) == 4:
                    clean_title = title[:title.rindex('(')].strip()
                    self.logger.debug(f"Successfully parsed: Title='{clean_title}', Year='{year_part}'")
                    return clean_title, year_part

            self.logger.warning(f"Could not extract year from title: {title}")
            self.stats['parsing_errors'] += 1
            return title.strip(), None

        except Exception as e:
            self.logger.error(f"Error parsing title '{title}': {str(e)}", exc_info=True)
            self.stats['parsing_errors'] += 1
            return title.strip(), None

    def get_movie_data(self, clean_title: str, year: str, cache: shelve.Shelf) -> Optional[Dict[str, Any]]:
        """Fetch movie data from TMDb API"""
        cache_key = f"{clean_title}_{year}"
        
        self.logger.debug(f"Fetching data for movie: '{clean_title}' ({year})")
        
        if cache_key in cache:
            self.logger.debug(f"Cache hit for '{clean_title}' ({year})")
            self.stats['cache_hits'] += 1
            return cache[cache_key]

        try:
            # Search for movie
            search_results = self.make_request(
                'search/movie',
                params={'query': clean_title, 'year': year}
            )
            
            if not search_results or not search_results.get('results'):
                self.logger.warning(f"No TMDb results found for '{clean_title}' ({year})")
                self.stats['failed_fetches'] += 1
                return None

            # Find best match
            matching_results = [
                movie for movie in search_results['results']
                if movie.get('release_date', '').startswith(year)
            ]

            if not matching_results:
                self.logger.warning(f"No results matching year {year} for '{clean_title}'")
                self.stats['failed_fetches'] += 1
                return None

            movie = matching_results[0]
            movie_id = movie['id']

            # Fetch detailed information
            movie_detail = self.make_request(f'movie/{movie_id}')
            credits = self.make_request(f'movie/{movie_id}/credits')
            external_ids = self.make_request(f'movie/{movie_id}/external_ids')

            if not all([movie_detail, credits, external_ids]):
                self.logger.error(f"Failed to fetch complete data for movie ID: {movie_id}")
                self.stats['failed_fetches'] += 1
                return None

            # Process movie data
            movie_data = {
                "Title": movie_detail['title'],
                "Original_Title": movie_detail['original_title'],
                "Year": year,
                "Released": movie_detail['release_date'],
                "Runtime": f"{movie_detail['runtime']} min" if movie_detail.get('runtime') else None,
                "Genre": ", ".join(genre['name'] for genre in movie_detail.get('genres', [])),
                "Director": ", ".join(
                    member['name'] for member in credits.get('crew', [])
                    if member.get('job') == 'Director'
                ),
                "Writer": ", ".join(
                    member['name'] for member in credits.get('crew', [])
                    if member.get('department') == 'Writing'
                ),
                "Actors": ", ".join(
                    member['name'] for member in credits.get('cast', [])[:5]
                ),
                "Plot": movie_detail['overview'],
                "imdbID": external_ids.get('imdb_id'),
                "TMDb_ID": movie_id
            }

            # Download poster if available
            if DOWNLOAD_POSTER:
                if movie_detail.get('poster_path'):
                    poster_path = self.download_poster(movie_detail['poster_path'], external_ids['imdb_id'])
                if poster_path:
                    movie_data["Poster_Path"] = poster_path
                    self.stats['poster_downloads'] += 1

            cache[cache_key] = movie_data
            self.stats['successful_fetches'] += 1
            self.logger.info(f"Successfully fetched data for '{clean_title}' ({year})")
            return movie_data

        except Exception as e:
            self.logger.error(f"Error fetching data for '{clean_title}' ({year}): {str(e)}", exc_info=True)
            self.stats['api_errors'] += 1
            return None

    def download_poster(self, poster_path: str, imdb_id: str) -> Optional[str]:
        """Download movie poster in specified size"""
        if not poster_path or not imdb_id:
            self.logger.warning(f"Missing poster path or IMDb ID for poster download")
            return None

        try:
            poster_url = f"https://image.tmdb.org/t/p/{POSTER_SIZE}{poster_path}"
            self.logger.debug(f"Downloading poster from: {poster_url}")

            response = requests.get(poster_url, headers=self.headers, stream=True, timeout=10)
            response.raise_for_status()

            poster_filename = f"{imdb_id}_{POSTER_SIZE}.jpg"
            poster_file_path = os.path.join(POSTER_DIR, poster_filename)

            with open(poster_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            self.logger.debug(f"Successfully downloaded poster to: {poster_file_path}")
            return poster_file_path

        except Exception as e:
            self.logger.error(f"Failed to download poster for IMDb ID {imdb_id}: {str(e)}", exc_info=True)
            self.stats['poster_failures'] += 1
            return None

    def crawl_movielens_data(self):
        """Main crawler function"""
        start_time = time.time()
        self.logger.info("Starting MovieLens data crawl")
        
        try:
            if not os.path.exists(MOVIELENS_CSV_PATH):
                self.logger.critical(f"MovieLens data file not found: {MOVIELENS_CSV_PATH}")
                raise FileNotFoundError(f"MovieLens data file not found: {MOVIELENS_CSV_PATH}")

            # Read MovieLens data
            self.logger.info(f"Reading MovieLens data from: {MOVIELENS_CSV_PATH}")
            GENRES = (
                "unknown", "Action", "Adventure", "Animation", "Children's", 
                "Comedy", "Crime", "Documentary", "Drama", "Fantasy", 
                "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
                "Sci-Fi", "Thriller", "War", "Western"
            )
            genre_cols = [f'genre_{genre}' for genre in GENRES]
            movies_df = pd.read_csv(
                MOVIELENS_CSV_PATH,
                sep='|',
                names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_cols,
                encoding="ISO-8859-1",
                engine="python"
            )

            self.stats['total_movies'] = len(movies_df)
            self.logger.info(f"Found {self.stats['total_movies']} movies in MovieLens dataset")

            fetched_movies = []
            with shelve.open(os.path.join(BASE_DIR, 'api_cache')) as cache:
                for index, row in tqdm(movies_df.iterrows(), total=len(movies_df)):
                    self.logger.debug(f"Processing movie {index + 1}/{len(movies_df)}")
                    
                    clean_title, year = self.parse_movie_title(row['movie_title'])
                    if not year:
                        continue

                    movie_data = self.get_movie_data(clean_title, year, cache)
                    if movie_data:
                        movie_data['MovieLens_ID'] = row['movie_id']
                        fetched_movies.append(movie_data)
                    
                    time.sleep(REQUEST_DELAY)

            # Save results
            if fetched_movies:
                df = pd.DataFrame(fetched_movies)
                df.to_csv(OUTPUT_CSV_PATH, index=False)
                self.logger.info(f"Successfully saved {len(fetched_movies)} movies to {OUTPUT_CSV_PATH}")
            else:
                self.logger.error("No movies were fetched successfully")

            # Log final statistics
            end_time = time.time()
            self.logger.info("\nCrawler Statistics:")
            self.logger.info(f"Total Runtime: {(end_time - start_time):.2f} seconds")
            self.logger.info(f"Total Movies Processed: {self.stats['total_movies']}")
            self.logger.info(f"Successful Fetches: {self.stats['successful_fetches']}")
            self.logger.info(f"Failed Fetches: {self.stats['failed_fetches']}")
            self.logger.info(f"Cache Hits: {self.stats['cache_hits']}")
            self.logger.info(f"Parsing Errors: {self.stats['parsing_errors']}")
            self.logger.info(f"API Errors: {self.stats['api_errors']}")
            self.logger.info(f"Poster Downloads: {self.stats['poster_downloads']}")
            self.logger.info(f"Poster Failures: {self.stats['poster_failures']}")

        except Exception as e:
            self.logger.critical(f"Critical error during crawl: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    crawler = MovieDataCrawler()
    crawler.crawl_movielens_data()