import heapq

import pandas as pd
import streamlit as st


# 세 개의 데이터 파일을 불러옵니다.
@st.cache_data
def load_data():
    edges_df = pd.read_csv('./edges_df.csv.gz')
    edges_undirected_df = pd.read_csv('./edges_undirected_df.csv.gz')
    s_id_title_series = pd.read_csv(
        './s_id_title.csv.gz', keep_default_na=False)['title']
    s_title_id_series = s_id_title_series.reset_index().set_index(
        'title', verify_integrity=True)['index'].sort_index()
    return edges_df, edges_undirected_df, s_id_title_series, s_title_id_series


edges_df, edges_undirected_df, s_id_title, s_title_id = load_data()


# from_title열에서 이분탐색을 수행합니다. 정렬된 from_title열을 기준으로 target과 일치하는 부분의 시작과 끝을 이분탐색으로 구합니다.
def find_lower_bound(target, _edges_df):
    left, right = 0, len(_edges_df)
    while left < right:
        mid = (left + right) // 2
        if _edges_df.iloc[mid]['from_title'] < target:
            left = mid + 1
        else:
            right = mid
    return left


def find_upper_bound(target, _edges_df):
    left, right = 0, len(_edges_df)
    while left < right:
        mid = (left + right) // 2
        if _edges_df.iloc[mid]['from_title'] <= target:
            left = mid + 1
        else:
            right = mid
    return left


def binary_search_title(target, _edges_df):
    """
    Perform binary search on a DataFrame sorted by 'from_title' to find all rows where from_title == target.

    Parameters:
        target (int): The title id to search for.

    Returns:
        pd.DataFrame: Subset of rows with from_title == target. Empty if not found.
    """

    lower = find_lower_bound(target, _edges_df)
    upper = find_upper_bound(target, _edges_df)

    if lower == upper:
        return pd.DataFrame(columns=_edges_df.columns)  # target not found

    return _edges_df.iloc[lower:upper]


# 문서 A -> B 최단 경로를 찾는 BFS 알고리즘
# BFS 알고리즘을 활용하여 나무위키 문서 A에서 문서 B로 링크를 타고 이동하는 최단 경로를 찾습니다.
# 문서 제목을 이분탐색으로 찾습니다.
# 최단 거리 경로가 여러 개이면 모두 구합니다.
# undirected가 True이면 간선이 두 방향 모두 존재하는 경우만을 고려합니다.
def bfs_shortest_path(start_title, target_title, undirected, progress_bar, status_text):
    """
    Finds the shortest path between two documents using BFS (layer-by-layer).

    Parameters:
        start_title (str): Title of the start document.
        target_title (str): Title of the target document.
        undirected (bool): If True, treat the graph as undirected.

    Returns:
        List[str] or None: List of titles from start to target, or None if no path exists.
    """
    start_id = s_title_id.get(start_title)
    target_id = s_title_id.get(target_title)
    if pd.isna(start_id) or pd.isna(target_id):
        return None

    if start_id == target_id:
        return [start_id]

    visited = set([start_id])
    queue = [[start_id]]
    result = []
    distance = 0

    _edges_df = edges_undirected_df if undirected else edges_df

    while queue:
        next_queue = []
        distance += 1

        for i, path in enumerate(queue):
            progress_bar.progress((i+1)/len(queue))
            status_text.text(f"searching for {distance=}: {i+1}/{len(queue)}")

            if len(result) * (len(queue)-i) > 1e5:
                break

            edges = binary_search_title(path[-1], _edges_df)

            for row in edges.itertuples(index=False):
                neighbor = row.to_title
                if neighbor in visited:
                    continue

                new_path = path + [neighbor]

                if neighbor == target_id:
                    result.append(new_path)
                else:
                    visited.add(neighbor)
                    next_queue.append(new_path)
        if result:
            return [list(map(lambda x: s_id_title.loc[x], path)) for path in result]
        # Move to next BFS layer
        queue = next_queue

    return None  # No path found


# 문서 A -> B 최단 경로를 찾는 다익스트라 알고리즘
# Dijkstra 알고리즘을 활용하여 나무위키 문서 A에서 문서 B로 링크를 타고 이동하는 최단 경로를 찾습니다.
# 문서 제목을 이분탐색으로 찾습니다.
# 가중치는 링크 개수의 역수이고, 양방향일 경우 두 방향의 링크 개수 역수의 평균입니다.
# undirected가 True이면 간선이 두 방향 모두 존재하는 경우만을 고려합니다.
def dijkstra_shortest_path(start_title, target_title, undirected, progress_bar, status_text):
    """
    Finds the shortest path between two documents using Dijkstra's algorithm.
    Edge weights are defined as 1 / count.

    Parameters:
        start_title (str): Title of the start document.
        target_title (str): Title of the target document.

    Returns:
        List[str] or None: List of titles from start to target, or None if no path exists.
    """
    start_id = s_title_id.get(start_title)
    target_id = s_title_id.get(target_title)
    if pd.isna(start_id) or pd.isna(target_id):
        return None

    # Priority queue: (cost, current_node, path, weights)
    heap = [(0, start_id, [start_id], [])]
    visited = set()

    _edges_df = edges_undirected_df if undirected else edges_df

    i = 0
    while heap:
        cost, node, path, weights = heapq.heappop(heap)
        if node == target_id:
            return cost, [s_id_title.loc[pid] for pid in path], weights

        if node in visited:
            continue
        visited.add(node)

        i += 1
        progress_bar.progress((i)/len(s_id_title))
        status_text.text(f"searching for nodes: {i}/{len(s_id_title)}")

        edges = binary_search_title(node, _edges_df)
        for row in edges.itertuples(index=False):
            neighbor = row.to_title
            weight = row.weight if undirected else 1/row.count
            if neighbor not in visited:
                heapq.heappush(
                    heap, (cost + weight, neighbor, path + [neighbor], weights+[weight]))

    return None  # No path found


# 제목 부분
st.title("Graph Calculation App")
st.header("Select Source and Target")

# 문서 입력 칸 부분
source = st.text_input("출발 문서의 제목을 입력하세요:", "")
target = st.text_input("도착 문서의 제목을 입력하세요:", "")

# Dijkstra/BFS, Directed/Undirected를 선택하는 체크박스 부분
weighted = st.checkbox("Do you want to use weighted graph?", value=True)
directed = st.checkbox("Do you want to use directed graph?", value=True)

# 계산 시작 버튼
execute_button = st.button("Execute Calculation")

# 버튼이 눌리면 실행됩니다
if execute_button:
    # 입력 값 유효성 검사
    if source not in s_title_id:
        st.error("출발 문서가 존재하지 않습니다.")
    elif target not in s_title_id:
        st.error("도착 문서가 존재하지 않습니다.")
    else:
        st.info(
            f"Starting calculation using {"Dijkstra" if weighted else "BFS"} with Source={source}, Target={target}, Directed={directed}")

        # 프로그레스바 표시 부분
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 체크박스에 따라 알고리즘 선택
        algorithm = dijkstra_shortest_path if weighted else bfs_shortest_path
        result = algorithm(source, target, not directed,
                           progress_bar, status_text)

        st.success("Calculation completed!")
        st.subheader("Result")

        if result:
            if weighted:
                cost, path, weights = result

                # 전체 경로 길이 표시
                st.markdown(f"**Total Cost:** `{cost:.2f}`")

                # 경로 표시
                st.markdown(f"**Path:**")
                # " node --(1.5)>> node " 형식으로 표시
                weighted_path_str = ""
                for j in range(len(path) - 2):
                    weighted_path_str += f"{path[j]} --`({weights[j]:.2f})`>> "
                weighted_path_str += path[-1]
                st.markdown(weighted_path_str)
            else:
                st.write("Shortest path(s) found:")
                # 발견한 경로를 모두 표시
                for i, path in enumerate(result):
                    st.write(f"Path {i+1}:")
                    st.markdown(" -> ".join(path))
        else:
            st.warning("No path found between the source and target documents.")
