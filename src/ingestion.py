import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices.
        
        Args:
            all_reports_dir (Path): Directory containing the JSON report files
            output_dir (Path): Directory where to save the BM25 indices
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # Load the report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Extract text chunks and create BM25 index
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    # def __init__(self):
        # self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[float]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        
        if isinstance(text, list):
            # text_chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
            text_chunks = text
        else:
            text_chunks = [text]

        # embeddings = []
        # for chunk in text_chunks:
        #     response = self.llm.embeddings.create(input=chunk, model=model)
        #     embeddings.extend([embedding.embedding for embedding in response.data])
        
        # return embeddings
    
        try:
            import requests
            import time
            url = "https://api.siliconflow.cn/v1/embeddings"
            model = "BAAI/bge-m3"

            api_key = "sk-vgzdgwcrvyexymfdhgsqnmohqhefwabsvuvrpwkggkkezpxy"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # 根据API限制调整参数
            max_tokens_per_request = 8192  # 单次请求最大token数
            rpm_limit = 2000  # 每分钟请求数限制
            tpm_limit = 500000  # 每分钟token数限制
            dimension = 1024  # 向量维度
            # 计算安全阈值 (80%的限制)
            safe_rpm = int(rpm_limit * 0.8)
            safe_tpm = int(tpm_limit * 0.8)

            all_embeddings = []
            last_request_time = 0
            tokens_sent_this_minute = 0
            requests_sent_this_minute = 0
            minute_start_time = time.time()

            for i in range(0, len(text_chunks)):
                # 检查速率限制
                current_time = time.time()
                if current_time - minute_start_time > 60:
                    # 重置分钟计数器
                    minute_start_time = current_time
                    tokens_sent_this_minute = 0
                    requests_sent_this_minute = 0
                
                # 计算当前文本块的token数 (简化估算)
                chunk_token_count = len(text_chunks[i].split()) * 1.5  # 1.5是安全系数
                
                # 检查是否超过限制
                if (tokens_sent_this_minute + chunk_token_count > safe_tpm or
                    requests_sent_this_minute + 1 > safe_rpm):
                    # 等待到下一分钟
                    sleep_time = 60 - (current_time - minute_start_time) + 1
                    time.sleep(max(0, sleep_time))
                    minute_start_time = time.time()
                    tokens_sent_this_minute = 0
                    requests_sent_this_minute = 0
                
                # 准备请求
                payload = {
                    "model": model,
                    "input": [text_chunks[i]],  # 单条处理确保不超过token限制
                    "encoding_format": "float"
                }
                
                # 发送请求
                response = requests.post(url, json=payload, headers=headers)
                
                # 更新计数器
                tokens_sent_this_minute += chunk_token_count
                requests_sent_this_minute += 1
                last_request_time = current_time
                
                # 处理响应
                if response.status_code != 200:
                    error_msg = f"API请求失败: {response.status_code} - {response.text}"
                    raise RuntimeError(error_msg)
                
                try:
                    response_data = response.json()
                    if not response_data.get("data"):
                        raise ValueError("API返回数据格式异常")
                        
                    all_embeddings.extend([item["embedding"] for item in response_data["data"]])
                    
                except (ValueError, KeyError, AttributeError) as e:
                    print(f"解析API响应失败: {str(e)}")
                    raise RuntimeError("无效的API响应格式") from e
                    
            return all_embeddings
            
        except requests.exceptions.RequestException as e:
            print(f"网络请求异常: {str(e)}")
            raise
        except Exception as e:
            print(f"获取嵌入向量时发生意外错误: {str(e)}")
            raise


    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")