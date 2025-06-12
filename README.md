# 2025 1학기 컴퓨팅 핵심 프로젝트

그래프 탐색 알고리즘과 6단계 분리 이론: 나무위키 문서 기반 최단 경로 탐색

## 파일 구조

데이터 파일은 `.csv.gz` 확장자를 가진 세 파일로 구성됩니다. gzip 압축을 해제하지 않아야 합니다.
- `edges_df.csv.gz`
- `edges_undirected_df.csv.gz`
- `s_id_title.csv.gz`

코드 파일은 `data_process.ipynb` 파일과 `graph_calculation_app.py`로 구성됩니다. 
ipynb 파일은 나무위키 덤프 데이터를 처리하여 상기한 세 그래프 데이터 파일로 변환하는 코드를 포함합니다. 코드를 실행한 뒤 만들어진 세 csv 파일을 gzip을 통해 압축하면 상기한 세 파일을 얻을 수 있습니다. 
py 파일은 그래프 데이터 파일을 사용하여 BFS, Dijkstra, 이분 탐색을 적용해 두 나무위키 문서 간 최단 경로를 계산하고 보여구는 `streamlit` 코드를 포함합니다. 

## 실행 방법

`data_process.ipynb` 파일은 코랩 환경에서 실행하였습니다. 필요한 라이브러리를 설치하는 `!pip install`문이 파일 맨 위에 있습니다. 

`graph_calculation_app.py`는 miniconda 환경에서 실행하였습니다. conda 환경 설정은 `environment.yml`에, 파이썬 패키지 설정은 `requirements.txt`에 있습니다. 주요한 설정과 버전은 다음과 같습니다. 
- python=3.12.11
- streamlit==1.45.1
- pandas==2.3.0

필요한 패키지를 설치한 후 다음 명령어를 실행하여 결과를 확인합니다. 
```bash
streamlit run graph_calculation_app.py
```

ngrok, docker(window)를 사용하여 외부에서 접속하게 하려면 `.env` 파일에 `NGROK_TOKEN`을 정의한 후 다음 명령을 실행합니다. 이 명령을 실행하지 않아도 위 `streamlit` 명령만으로 localhost에서 결과를 확인할 수 있습니다. 
```bash
docker run --net=host -it -e NGROK_AUTHTOKEN=$NGROK_TOKEN ngrok/ngrok http host.docker.internal:8501
```