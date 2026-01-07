# SERP 데이터로 GPT-4o를 사용해 RAG 챗봇 만들기

[![Promo](https://media.brightdata.com/2025/08/SERP-API-50-off-GitHub-banner_1389_166.png)](https://brightdata.co.kr/) 

이 가이드는 GPT-4o와 Bright Data의 SERP API를 사용하여 더 정확하고 컨텍스트가 풍부한 AI 응답을 제공하는 Python RAG 챗봇을 구축하는 방법을 설명합니다.

1. [소개](#how-to-creating-a-rag-chatbot-with-gpt-4o-using-serp-data)
2. [RAG란 무엇입니까?](#what-is-rag)
3. [왜 AI 모델에 SERP 데이터를 제공해야 합니까](#why-feed-ai-models-with-serp-data)
4. [Python으로 GPT 모델과 SERP 데이터를 사용하는 RAG: 단계별 튜토리얼](#rag-with-serp-data-with-gpt-models-using-python-step-by-step-tutorial)
    1. [Step #1: Python 프로젝트 초기화](#step-1-initialize-a-python-project)
    2. [Step #2: 필요한 라이브러리 설치](#step-2-install-the-required-libraries)
    3. [Step #3: 프로젝트 준비](#step-3-prepare-your-project)
    4. [Step #4: SERP API 구성](#step-4-configure-serp-api)
    5. [Step #5: SERP スクレイピング 로직 구현](#step-5-implement-the-serp-scraping-logic)
    6. [Step #6: SERP URL에서 텍스트 추출](#step-6-extract-text-from-the-serp-urls)
    7. [Step #7: RAG 프롬프트 생성](#step-7-generate-the-rag-prompt)
    8. [Step #8: GPT 요청 수행](#step-8-perform-the-gpt-request)
    9. [Step #9: 애플리케이션 UI 생성](#step-9-create-the-application-ui)
    10. [Step #10: 모두 통합](#step-10-put-it-all-together)
    11. [Step #11: 애플리케이션 테스트](#step-11-test-the-application)
5. [결론](#conclusion)

## What Is RAG?

RAG는 [Retrieval-Augmented Generation](https://blogs.nvidia.comhttps://brightdata.co.kr/blog/what-is-retrieval-augmented-generation/)의 약자로, 정보 검색과 텍스트 생성을 결합하는 AI 접근 방식입니다. RAG 워크플로우에서는 애플리케이션이 먼저 문서, 웹 페이지, 데이터베이스 등과 같은 외부 소스에서 관련 데이터를 검색합니다. 그런 다음 해당 데이터를 AI 모델에 전달하여 더 문맥적으로 관련성 높은 응답을 생성할 수 있도록 합니다.

RAG는 GPT와 같은 대규모 언어 모델(LLM)이 원래의 학습 데이터 너머의 최신 정보에 접근하고 참조할 수 있도록 함으로써 성능을 강화합니다. 이 접근 방식은 정확하고 컨텍스트에 특화된 정보가 필요한 시나리오에서 핵심이며, AI가 생성한 응답의 품질과 정확도를 모두 향상시킵니다.

## Why Feed AI Models With SERP Data

GPT-4o의 지식 컷오프 날짜는 [2023년 10월](https://computercity.com/artificial-intelligence/knowledge-cutoff-dates-llms)로, 그 이후에 공개된 사건이나 정보에는 접근할 수 없음을 의미합니다. 하지만 [GPT-4o models](https://openai.com/index/hello-gpt-4o/)는 Bing 검색 통합을 사용해 실시간으로 Internet에서 데이터를 가져올 수 있습니다. 이를 통해 더 최신의 정보를 제공하고, 상세하며 정확하고 컨텍스트가 풍부한 응답을 제시할 수 있습니다.

## RAG With SERP Data With GPT Models Using Python: Step-By-Step Tutorial

이 튜토리얼은 OpenAI의 GPT 모델을 사용하여 RAG 챗봇을 구축하는 과정을 안내합니다. 아이디어는 특정 검색 쿼리에 대해 Google에서 상위 성과를 내는 페이지들로부터 텍스트를 수집하고, 이를 GPT 요청의 컨텍스트로 사용하는 것입니다.

가장 큰 과제는 SERP 데이터의 スクレイピング입니다. 대부분의 검색 엔진은 페이지에 대한 자동화된 접근을 방지하기 위해 고급 アンチボット 솔루션을 갖추고 있습니다. 자세한 안내는 [how to scrape Google in Python](https://brightdata.co.kr/blog/web-data/scraping-google-with-python) 가이드를 참고하십시오.

スクレイピング 프로세스를 단순화하기 위해 [Bright Data’s SERP API](https://brightdata.co.kr/products/serp-api)를 사용하겠습니다.

이 SERP スクレイ퍼를 사용하면 단순한 HTTP 요청으로 Google, DuckDuckGo, Bing, Yandex, Baidu 및 기타 검색 엔진의 SERP를 손쉽게 가져올 수 있습니다.

그런 다음 [headless browser](https://brightdata.co.kr/blog/web-data/best-headless-browsers)를 사용하여 반환된 URL에서 텍스트 데이터를 추출하겠습니다. 이후 해당 정보를 RAG 워크플로우에서 GPT 모델의 컨텍스트로 사용하겠습니다. 대신 AI를 사용하여 온라인 데이터를 직접 검색해 가져오고 싶다면, [web scraping with ChatGPT](https://brightdata.co.kr/blog/web-data/web-scraping-with-chatgpt) 문서를 읽어보십시오.

이 가이드의 모든 코드는 GitHub 리포지토리에서도 확인할 수 있습니다:

```bash
git clone https://github.com/Tonel/rag_gpt_serp_scraping
```

README.md 파일의 지침에 따라 프로젝트의 의존성을 설치하고 프로젝트를 실행하십시오.

이 블로그 포스트에서 제시하는 접근 방식은 다른 어떤 검색 엔진이나 LLM에도 쉽게 적용할 수 있다는 점을 유념하십시오.

> **Note**:\
> 이 가이드는 Unix 및 macOS를 기준으로 합니다. Windows 사용자라면 [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)을 사용하여 튜토리얼을 그대로 따라 하실 수 있습니다.

### Step #1: Initialize a Python Project

머신에 Python 3가 설치되어 있는지 확인하십시오. 설치되어 있지 않다면 [다운로드하여 설치](https://www.python.org/downloads/)하십시오.

프로젝트용 폴더를 만들고 터미널에서 해당 폴더로 이동합니다:

```bash
mkdir rag_gpt_serp_scraping

cd rag_gpt_serp_scraping
```

`rag_gpt_serp_scraping` 폴더에는 Python RAG 프로젝트가 포함됩니다.

그다음 선호하는 Python IDE에서 프로젝트 디렉터리를 여십시오. [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/) 또는 [Visual Studio Code with the Python extension](https://code.visualstudio.com/docs/languages/python)을 사용하면 됩니다.

rag\_gpt\_serp\_scraping 내부에 빈 app.py 파일을 추가하십시오. 이 파일에는 スクレイピング 및 RAG 로직이 포함됩니다.

다음으로, 프로젝트 디렉터리에서 [Python virtual environment](https://docs.python.org/3/library/venv.html)를 초기화하십시오:

```bash
python3 -m venv env
```

아래 명령으로 virtual environment를 활성화하십시오:

```bash
source ./env/bin/activate
```

### Step #2: Install the Required Libraries

이 Python RAG 프로젝트는 다음 의존성을 사용합니다:

*   [`python-dotenv`](https://pypi.org/project/python-dotenv/): Bright Data 자격 증명 및 OpenAI API key와 같은 민감한 자격 증명을 안전하게 관리하는 데 사용합니다.
*   [`requests`](https://pypi.org/project/requests/): Bright Data의 SERP API에 HTTP 요청을 수행합니다.
*   [`langchain-community`](https://pypi.org/project/langchain-community/): Google SERP 페이지에서 텍스트를 가져오고 이를 정리하여 RAG에 적합한 콘텐츠를 생성하는 데 사용합니다.
*   [`openai`](https://pypi.org/project/openai/): GPT 모델과 연동하여 입력 및 RAG 컨텍스트에 기반한 자연어 응답을 생성하는 데 사용합니다.
*   [`streamlit`](https://pypi.org/project/streamlit/): 사용자가 Google 검색 쿼리와 AI 프롬프트를 입력하고, 결과를 동적으로 확인할 수 있는 UI를 만드는 데 유용합니다.

모든 의존성을 설치하십시오:

```bash
pip install python-dotenv requests langchain-community openai streamlit
```

langchain-community의 [AsyncChromiumLoader](https://python.langchain.com/docs/integrations/document_loaders/async_chromium/)를 사용할 예정이며, 이는 다음 의존성을 필요로 합니다:

```bash
pip install --upgrade --quiet playwright beautifulsoup4 html2text
```

정상적으로 동작하려면 Playwright에서 브라우저 설치도 필요합니다:

```bash
playwright install
```

### Step #3: Prepare Your Project

`app.py`에 다음 import를 추가하십시오:

```python
from dotenv import load_dotenv

import os

import requests

from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_community.document_transformers import BeautifulSoupTransformer

from openai import OpenAI

import streamlit as st
```

그다음 프로젝트 폴더에 `.env` 파일을 만들어 모든 자격 증명을 저장하십시오. 이제 프로젝트 구조는 아래와 같이 보입니다:

![Project structure](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-19.png)

`app.py`에서 아래 함수를 사용하여 `python-dotenv`가 `.env`에서 environment variables를 로드하도록 지시하십시오:

```python
load_dotenv()
```

이제 `.env` 또는 시스템에서 environment variables를 다음과 같이 가져올 수 있습니다:

```python
os.environ.get("<ENV_NAME>")
```

### Step #4: Configure SERP API

Bright Data의 SERP API를 사용하여 검색 엔진 결과 페이지의 콘텐츠를 가져오고 이를 Python RAG 워크플로우에 사용하겠습니다. 구체적으로는 SERP API가 반환하는 웹 페이지 URL에서 텍스트를 추출할 것입니다.

SERP API를 설정하려면 [official documentation](https://docs.brightdata.com/scraping-automation/serp-api/quickstart)를 참고하십시오. 또는 아래 지침을 따르십시오.

아직 계정을 만들지 않았다면 [Bright Data에 가입](https://brightdata.co.kr)하십시오. 로그인한 뒤 계정 대시보드로 이동합니다:

![Account main dashboard](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-18.png)

거기서 “Get proxy products” 버튼을 클릭하십시오.

그러면 아래 페이지로 이동하며, “SERP API” 행을 클릭해야 합니다:

![Clicking on SERP API](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-17.png)

SERP API 제품 페이지에서 “Activate zone”을 토글하여 제품을 활성화하십시오:

![Activating the SERP zone](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-16.png)

이제 “Access parameters” 섹션에서 SERP API host, port, username, password를 복사하여 `.env` 파일에 추가하십시오:

```python
BRIGHT_DATA_SERP_API_HOST="<YOUR_HOST>"

BRIGHT_DATA_SERP_API_PORT=<YOUR_PORT>

BRIGHT_DATA_SERP_API_USERNAME="<YOUR_USERNAME>"

BRIGHT_DATA_SERP_API_PASSWORD="<YOUR_PASSWORD>"
```

`<YOUR_XXXX>` 플레이스홀더를 SERP API 페이지에서 Bright Data가 제공한 값으로 교체하십시오.

“Access parameters”의 host는 다음과 같은 형식일 수 있습니다:

```python
brd.superproxy.io:33335
```

이를 아래처럼 분리하십시오:

```python
BRIGHT_DATA_SERP_API_HOST="brd.superproxy.io"

BRIGHT_DATA_SERP_API_PORT=33335
```

### Step #5: Implement the SERP Scraping Logic

`app.py`에 다음 함수를 추가하여 Google SERP 페이지에서 첫 번째 `number_of_urls` 개 URL을 가져오십시오:

```python
def get_google_serp_urls(query, number_of_urls=5):

# perform a Bright Data's SERP API request

# with JSON autoparsing

host = os.environ.get("BRIGHT_DATA_SERP_API_HOST")

port = os.environ.get("BRIGHT_DATA_SERP_API_PORT")

username = os.environ.get("BRIGHT_DATA_SERP_API_USERNAME")

password = os.environ.get("BRIGHT_DATA_SERP_API_PASSWORD")

proxy_url = f"http://{username}:{password}@{host}:{port}"

proxies = {"http": proxy_url, "https": proxy_url}

url = f"https://www.google.com/search?q={query}&brd_json=1"

response = requests.get(url, proxies=proxies, verify=False)

# retrieve the parsed JSON response

response_data = response.json()

# extract a "number_of_urls" number of

# Google SERP URLs from the response

google_serp_urls = []

if "organic" in response_data:

for item in response_data["organic"]:

if "link" in item:

google_serp_urls.append(item["link"])

return google_serp_urls[:number_of_urls]
```

이는 query 인수에 지정된 검색 쿼리로 SERP API에 HTTP GET 요청을 보냅니다. [`brd_json=1`](https://docs.brightdata.com/scraping-automation/serp-api/parsing-search-results) 쿼리 파라メータ는 SERP API가 결과를 아래 형식의 JSON으로 파싱하도록 보장합니다:

```json
{

"general": {

"search_engine": "google",

"results_cnt": 1980000000,

"search_time": 0.57,

"language": "en",

"mobile": false,

"basic_view": false,

"search_type": "text",

"page_title": "pizza - Google Search",

"code_version": "1.90",

"timestamp": "2023-06-30T08:58:41.786Z"

},

"input": {

"original_url": "https://www.google.com/search?q=pizza&brd_json=1",

"user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12) AppleWebKit/608.2.11 (KHTML, like Gecko) Version/13.0.3 Safari/608.2.11",

"request_id": "hl_1a1be908_i00lwqqxt1"

},

"organic": [

{

"link": "https://www.pizzahut.com/",

"display_link": "https://www.pizzahut.com",

"title": "Pizza Hut | Delivery & Carryout - No One OutPizzas The Hut!",

"image": "omitted for brevity...",

"image_alt": "pizza from www.pizzahut.com",

"image_base64": "omitted for brevity...",

"rank": 1,

"global_rank": 1

},

{

"link": "https://www.dominos.com/en/",

"display_link": "https://www.dominos.com › ...",

"title": "Domino's: Pizza Delivery & Carryout, Pasta, Chicken & More",

"description": "Order pizza, pasta, sandwiches & more online for carryout or delivery from Domino's. View menu, find locations, track orders. Sign up for Domino's email ...",

"image": "omitted for brevity...",

"image_alt": "pizza from www.dominos.com",

"image_base64": "omitted for brevity...",

"rank": 2,

"global_rank": 3

},

// omitted for brevity...

],

// omitted for brevity...

}
```

함수의 마지막 몇 줄은 결과 JSON 데이터에서 각 SERP URL을 가져오고, 첫 번째 `number_of_urls` 개 URL만 선택하여 리스트로 반환합니다.

### Step #6: Extract Text from the SERP URLs

각 SERP URL에서 텍스트를 추출하는 함수를 정의하십시오:

```python
# Note: Some websites may have dynamic content or anti-scraping measures that could prevent text extraction.
# In such cases, please consider using additional tools like Selenium
def extract_text_from_urls(urls, number_of_words=600): 

# instruct a headless Chrome instance to visit the provided URLs

# with the specified user-agent

loader = AsyncChromiumLoader(

urls,

user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",

)

html_documents = loader.load()

# process the extracted HTML documents to extract text from them

bs_transformer = BeautifulSoupTransformer()

docs_transformed = bs_transformer.transform_documents(

html_documents,

tags_to_extract=["p", "em", "li", "strong", "h1", "h2"],

unwanted_tags=["a"],

remove_comments=True,

)

# make sure each HTML text document contains only a number

# number_of_words words

extracted_text_list = []

for doc_transformed in docs_transformed:

# split the text into words and join the first number_of_words

words = doc_transformed.page_content.split()[:number_of_words]

extracted_text = " ".join(words)

# ignore empty text documents

if len(extracted_text) != 0:

extracted_text_list.append(extracted_text)

return extracted_text_list
```

이 함수는 다음을 수행합니다:

1.  headless Chrome browser 인스턴스를 사용하여 인수로 전달된 URL에서 웹 페이지를 로드합니다.
2.  [BeautifulSoupTransformer](https://python.langchain.com/v0.2/api_reference/community/document_transformers/langchain_community.document_transformers.beautiful_soup_transformer.BeautifulSoupTransformer.html)를 사용해 각 페이지의 HTML을 처리하고, 특정 태그(예: `<p>`, `<h1>`, `<strong>` 등)에서 텍스트를 추출하며, 불필요한 태그(예: `<a>`)와 주석을 제외합니다.
3.  각 웹 페이지에 대해 `number_of_words` 인수로 지정된 단어 수로 추출 텍스트를 제한합니다.
4.  각 URL에서 추출된 텍스트의 리스트를 반환합니다.

`["p", "em", "li", "strong", "h1", "h2"]` 태그는 대부분의 웹 페이지에서 텍스트를 추출하기에 충분하지만, 특정 시나리오에서는 이 HTML 태그 리스트를 커스터마이징해야 할 수 있습니다. 또한 각 텍스트 항목의 목표 단어 수를 늘리거나 줄여야 할 수도 있습니다.

예를 들어, 아래 [web page](https://athomeinhollywood.com/2024/09/19/transformers-one-review/)를 고려해 보십시오:

![Transformers one review page](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-15.png)

해당 페이지에 이 함수를 적용하면 다음 텍스트 배열이 생성됩니다:

```python
["Lisa Johnson Mandell’s Transformers One review reveals the heretofore inconceivable: It’s one of the best animated films of the year! I never thought I’d see myself write this about a Transformers movie, but Transformers One is actually an exceptional film! ..."]
```

`extract_text_from_urls()`가 반환하는 텍스트 항목 리스트는 OpenAI 모델에 제공할 RAG 컨텍스트를 의미합니다.

### Step #7: Generate the RAG Prompt

AI 프롬프트 요청과 텍스트 컨텍스트를 최종 RAG 프롬프트로 변환하는 함수를 정의하십시오:

```python
def get_openai_prompt(request, text_context=[]):

# default prompt

prompt = request

# add the context to the prompt, if present

if len(text_context) != 0:

context_string = "\n\n--------\n\n".join(text_context)

prompt = f"Answer the request using only the context below.\n\nContext:\n{context_string}\n\nRequest: {request}"

return prompt
```

RAG 컨텍스트가 지정된 경우, 이전 함수가 반환하는 프롬프트는 다음 형식입니다:

```
Answer the request using only the context below.

Context:

Bla bla bla...

--------

Bla bla bla...

--------

Bla bla bla...

Request: <YOUR_REQUEST>
```

### Step #8: Perform the GPT Request

먼저 `app.py` 파일 상단에서 OpenAI client를 초기화하십시오:

```python
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

이는 `OPENAI_API_KEY` environment variable에 의존하며, 이를 시스템 environment variables 또는 `.env` 파일에서 직접 정의할 수 있습니다:

`OPENAI_API_KEY="<YOUR_API_KEY>"`

`<YOUR_API_KEY>`를 [OpenAI API key](https://platform.openai.com/api-keys) 값으로 바꾸십시오. 발급 방법을 모른다면 [official guide](https://platform.openai.com/docs/quickstart)를 따르십시오.

다음으로 OpenAI official client를 사용해 [GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) AI 모델에 요청을 수행하는 함수를 작성하십시오:

```python
def interrogate_openai(prompt, max_tokens=800):

# interrogate the OpenAI model with the given prompt

response = openai_client.chat.completions.create(

model="gpt-4o-mini",

messages=[{"role": "user", "content": prompt}],

max_tokens=max_tokens,

)

return response.choices[0].message.content
```

> **Note**:\
> OpenAI API가 지원하는 다른 어떤 GPT 모델로도 구성할 수 있습니다.

지정된 텍스트 컨텍스트를 포함하는 `get_openai_prompt()`의 프롬프트로 호출하면, `interrogate_openai()`는 의도한 대로 retrieval-augmented generation을 성공적으로 수행합니다.

### Step #9: Create the Application UI

Streamlit을 사용하여 사용자가 다음을 지정할 수 있는 간단한 [form UI](https://docs.streamlit.io/develop/concepts/architecture/forms)를 정의하십시오:

1.  SERP API에 전달할 Google 검색 쿼리
2.  GPT-4o mini에 보낼 AI 프롬프트

이를 위해 다음 코드를 사용하십시오:

```python
with st.form("prompt_form"):

# initialize the output results

result = ""

final_prompt = ""

# textarea for user to input their Google search query

google_search_query = st.text_area("Google Search:", None)

# textarea for user to input their AI prompt

request = st.text_area("AI Prompt:", None)

# button to submit the form

submitted = st.form_submit_button("Send")

# if the form is submitted

if submitted:

# retrieve the Google SERP URLs from the given search query

google_serp_urls = get_google_serp_urls(google_search_query)

# extract the text from the respective HTML pages

extracted_text_list = extract_text_from_urls(google_serp_urls)

# generate the AI prompt using the extracted text as context

final_prompt = get_openai_prompt(request, extracted_text_list)

# interrogate an OpenAI model with the generated prompt

result = interrogate_openai(final_prompt)

# dropdown containing the generated prompt

final_prompt_expander = st.expander("AI Final Prompt:")

final_prompt_expander.write(final_prompt)

# write the result from the OpenAI model

st.write(result)
```

이제 Python RAG 스크립트가 준비되었습니다.

### Step #10: Put It All Together

`app.py` 파일에는 다음 코드가 포함되어야 합니다:

```python
from dotenv import load_dotenv

import os

import requests

from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_community.document_transformers import BeautifulSoupTransformer

from openai import OpenAI

import streamlit as st

# load the environment variables from the .env file

load_dotenv()

# initialize the OpenAI API client with your API key

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_google_serp_urls(query, number_of_urls=5):

# perform a Bright Data's SERP API request

# with JSON autoparsing

host = os.environ.get("BRIGHT_DATA_SERP_API_HOST")

port = os.environ.get("BRIGHT_DATA_SERP_API_PORT")

username = os.environ.get("BRIGHT_DATA_SERP_API_USERNAME")

password = os.environ.get("BRIGHT_DATA_SERP_API_PASSWORD")

proxy_url = f"http://{username}:{password}@{host}:{port}"

proxies = {"http": proxy_url, "https": proxy_url}

url = f"https://www.google.com/search?q={query}&brd_json=1"

response = requests.get(url, proxies=proxies, verify=False)

# retrieve the parsed JSON response

response_data = response.json()

# extract a "number_of_urls" number of

# Google SERP URLs from the response

google_serp_urls = []

if "organic" in response_data:

for item in response_data["organic"]:

if "link" in item:

google_serp_urls.append(item["link"])

return google_serp_urls[:number_of_urls]

def extract_text_from_urls(urls, number_of_words=600):

# instruct a headless Chrome instance to visit the provided URLs

# with the specified user-agent

loader = AsyncChromiumLoader(

urls,

user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",

)

html_documents = loader.load()

# process the extracted HTML documents to extract text from them

bs_transformer = BeautifulSoupTransformer()

docs_transformed = bs_transformer.transform_documents(

html_documents,

tags_to_extract=["p", "em", "li", "strong", "h1", "h2"],

unwanted_tags=["a"],

remove_comments=True,

)

# make sure each HTML text document contains only a number

# number_of_words words

extracted_text_list = []

for doc_transformed in docs_transformed:

# split the text into words and join the first number_of_words

words = doc_transformed.page_content.split()[:number_of_words]

extracted_text = " ".join(words)

# ignore empty text documents

if len(extracted_text) != 0:

extracted_text_list.append(extracted_text)

return extracted_text_list

def get_openai_prompt(request, text_context=[]):

# default prompt

prompt = request

# add the context to the prompt, if present

if len(text_context) != 0:

context_string = "\n\n--------\n\n".join(text_context)

prompt = f"Answer the request using only the context below.\n\nContext:\n{context_string}\n\nRequest: {request}"

return prompt

def interrogate_openai(prompt, max_tokens=800):

# interrogate the OpenAI model with the given prompt

response = openai_client.chat.completions.create(

model="gpt-4o-mini",

messages=[{"role": "user", "content": prompt}],

max_tokens=max_tokens,

)

return response.choices[0].message.content

# create a form in the Streamlit app for user input

with st.form("prompt_form"):

# initialize the output results

result = ""

final_prompt = ""

# textarea for user to input their Google search query

google_search_query = st.text_area("Google Search:", None)

# textarea for user to input their AI prompt

request = st.text_area("AI Prompt:", None)

# button to submit the form

submitted = st.form_submit_button("Send")

# if the form is submitted

if submitted:

# retrieve the Google SERP URLs from the given search query

google_serp_urls = get_google_serp_urls(google_search_query)

# extract the text from the respective HTML pages

extracted_text_list = extract_text_from_urls(google_serp_urls)

# generate the AI prompt using the extracted text as context

final_prompt = get_openai_prompt(request, extracted_text_list)

# interrogate an OpenAI model with the generated prompt

result = interrogate_openai(final_prompt)

# dropdown containing the generated prompt

final_prompt_expander = st.expander("AI Final Prompt")

final_prompt_expander.write(final_prompt)

# write the result from the OpenAI model

st.write(result)
```

### Step #11: Test the Application

다음으로 Python RAG 애플리케이션을 실행하십시오:

```bash
# Note: Streamlit is designed for lightweight applications. For production-grade deployments, consider using frameworks like Flask or FastAPI.
streamlit run app.py
```
터미널에서 다음 출력이 표시되어야 합니다:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501

Network URL: http://172.27.134.248:8501
```

지침에 따라 브라우저에서 `http://localhost:8501`로 이동하십시오. 아래와 같은 화면이 표시될 것입니다:

![Streamlit app screenshot](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-14.png)

아래와 같은 Google 검색 쿼리를 사용하여 애플리케이션을 테스트하십시오:

```
Transformers One review
```

그리고 다음과 같은 AI 프롬프트를 입력하십시오:

```
Write a review for the movie Transformers One
```

“Send”를 클릭하고 애플리케이션이 요청을 처리할 때까지 기다리십시오. 몇 초 후 아래와 같은 결과를 얻을 수 있습니다:

![App result screenshot](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-13.png)

“AI Final Prompt” 드롭다운을 펼치면, 애플리케이션이 RAG를 위해 사용한 전체 프롬프트를 확인할 수 있습니다.

## Conclusion

Python RAG 챗봇을 사용할 때의 주요 과제는 Google과 같은 검색 엔진을 スクレイピング하는 것입니다:

1. SERP 페이지 구조를 자주 변경합니다.
2. 사용 가능한 アンチボット 대책 중에서도 가장 정교한 수준의 보호가 적용되어 있습니다.
3. 대량의 SERP 데이터를 同時接続로 가져오는 것은 복잡하며 비용이 많이 들 수 있습니다.

[Bright Data’s SERP API](https://brightdata.co.kr/products/serp-api)는 주요 검색 엔진의 실시간 SERP 데이터를 노력 없이 가져올 수 있도록 도와드립니다. 또한 RAG 및 다양한 다른 애플리케이션도 지원합니다. 지금 무료 체험을 시작해 보십시오!