import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
import tqdm


class Spider:
    def __init__(self):
        chrome_opts = webdriver.ChromeOptions()
        chrome_opts.add_argument("--headless")
        chrome_opts.add_argument("--no-sandbox")
        chrome_opts.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())

        self.driver = webdriver.Chrome(service=service, options=chrome_opts)
        self.driver.implicitly_wait(10)

    def scrape_links(self, professor_dfs: list, limit=2000, num_of_references=10):
        count = 0
        new_dfs = []
        entities = []

        with tqdm.tqdm(total=limit) as pbar:
            while count < limit:
                for professor_df in professor_dfs:
                    pbar.set_description(
                        f"Scraping {professor_df['related_prof'][0]}'s papers"
                    )
                    urls_to_scrape = professor_df[~professor_df["is_crawled"]][
                        "link"
                    ].tolist()

                    for url in urls_to_scrape:
                        if count >= limit:
                            return professor_dfs

                        try:
                            time.sleep(1)
                            self.driver.get(url)
                            soup = BeautifulSoup(self.driver.page_source, "html.parser")

                            references = soup.find("div", {"data-test-id": "reference"})
                            references = references.find(
                                "div", {"class": "citation-list__citations"}
                            )
                            rows = references.find_all(
                                "div",
                                {"class": "cl-paper-row citation-list__paper-row"},
                            )

                            title = soup.find(
                                "h1", {"data-test-id": "paper-detail-title"}
                            ).text
                            authors = soup.find("span", {"class": "author-list"})
                            authors = authors.find_all(
                                "span", {"data-heap-id": "heap-author-list-item"}
                            )
                            year = soup.find("span", {"data-test-id": "paper-year"})
                            abstract = soup.find(
                                "div", {"data-test-id": "paper-abstract"}
                            ).text

                            topics = soup.find(
                                "ul", {"class": "flex-row-vcenter paper-meta"}
                            )

                            references = references.find_all(
                                "a", {"class": "link-button--show-visited"}
                            )
                            new_links = []

                            for reference in references[:num_of_references]:
                                link = reference["href"]
                                if (
                                    link not in professor_df["link"].values
                                    and link not in new_links
                                ):
                                    new_links.append(link)
                                    print(link)
                                    count += 1
                                    pbar.update(1)

                            if len(new_links) > 0:
                                new_df = pd.DataFrame(
                                    {
                                        "link": new_links,
                                        "is_crawled": False,
                                        "related_prof": professor_df["related_prof"][0],
                                    }
                                )
                                new_entity = {
                                    "URL": link,
                                    "Title": title,
                                    "Authors": authors,
                                    "Abstract": abstract,
                                }

                                entities.append(new_entity)
                                professor_df = pd.concat([professor_df, new_df])
                                professor_df.reset_index(drop=True, inplace=True)

                            professor_df.loc[
                                professor_df["link"] == url, "is_crawled"
                            ] = True

                            new_dfs.append(professor_df)

                        except Exception as e:
                            print(e)
                            continue
            return new_dfs, entities