# -*- coding: utf-8 -*-
"""Автотесты RAG: нарезка, очистка PDF-колонтитулов, косинус, качество поиска.

Гоняются offline, без LM Studio: эмбеддинги подменяются детерминированным
фейком (bag-of-words), которого хватает, чтобы проверить, что поиск достаёт
правильную статью. Запуск из корня проекта:

    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag  # noqa: E402


# ── фейковый клиент эмбеддингов ──────────────────────────────────────────────
# Вектор = bag-of-words: у похожих по словам текстов косинус близок к 1.
# Хватает, чтобы проверить пайплайн поиска без сервера. crc32 — ради
# детерминизма (обычный hash() рандомизирован PYTHONHASHSEED).

_DIM = 512


def _fake_vector(text: str) -> list[float]:
    import re

    vec = [0.0] * _DIM
    for tok in re.findall(r"\w+", text.lower()):
        vec[zlib.crc32(tok.encode("utf-8")) % _DIM] += 1.0
    return vec


class _FakeEmbeddings:
    def create(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=_fake_vector(t)) for t in input])


class FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


class _FakeChat:
    def __init__(self, reply):
        self._reply = reply

    def create(self, model, messages, temperature, max_tokens):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._reply))])


class FakeChatClient:
    """Чат-клиент с заранее заданным ответом — для rewriter/reranker без LM Studio."""

    def __init__(self, reply):
        self.chat = SimpleNamespace(completions=_FakeChat(reply))


# ── чистая логика нарезки ────────────────────────────────────────────────────
class TestChunkText(unittest.TestCase):
    def test_size_and_overlap(self):
        text = "abcdefghij" * 30  # 300 символов
        chunks = rag.chunk_text(text, size=100, overlap=20)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        # соседние окна перекрываются: хвост первого = начало второго
        self.assertEqual(chunks[0][-20:], chunks[1][:20])

    def test_strips_and_drops_empty(self):
        self.assertEqual(rag.chunk_text("   \n  \t  "), [])
        self.assertEqual(rag.chunk_text("  привет  ", size=100, overlap=20), ["привет"])

    def test_no_infinite_loop_on_bad_overlap(self):
        # overlap >= size не должен зацикливать (step ограничен снизу единицей -> шаг по 1)
        self.assertEqual(rag.chunk_text("абвгд", size=2, overlap=5),
                         ["аб", "бв", "вг", "гд", "д"])


class TestStructuredChunks(unittest.TestCase):
    def test_none_when_no_articles(self):
        self.assertIsNone(rag.structured_chunks("обычный текст без всяких статей"))

    def test_splits_by_article_and_keeps_header(self):
        text = (
            "Раздел I. Общие положения\n"
            "Статья 1. Цели. Установление гарантий трудовых прав.\n"
            "Статья 2. Принципы. Свобода труда и запрет дискриминации."
        )
        chunks = rag.structured_chunks(text)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Статья 1."))
        self.assertTrue(chunks[1].startswith("Статья 2."))
        # преамбула до первой статьи в чанки не попадает
        self.assertFalse(any("Раздел I" in c for c in chunks))

    def test_sub_numbered_article(self):
        text = "Статья 351.1. Специальная норма. Текст нормы."
        chunks = rag.structured_chunks(text)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].startswith("Статья 351.1."))

    def test_long_article_repeats_header_in_every_piece(self):
        # ключевой инвариант: длинную статью режем окном, но заголовок
        # повторяется в каждом куске — контекст не теряет привязку
        body = "норма и её подробное описание. " * 120  # > ARTICLE_MAX
        text = "Статья 100. Заголовок. " + body
        chunks = rag.structured_chunks(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.startswith("Статья 100.") for c in chunks))


class TestSectionOf(unittest.TestCase):
    def test_article_label(self):
        self.assertEqual(rag.section_of("Статья 228. Обязанности работодателя ..."), "Статья 228.")
        self.assertEqual(rag.section_of("Статья 351.1. Особенности ..."), "Статья 351.1.")

    def test_fallback_prefix_for_non_legal(self):
        self.assertTrue(rag.section_of("Просто произвольный текст без статьи").endswith("…"))


# ── очистка колонтитулов КонсультантПлюс (проверка качества PDF) ──────────────
class TestStripBoilerplate(unittest.TestCase):
    # блок ровно в том виде, в каком pypdf выдаёт его на границе страницы
    RAW = (
        "предметом дальнейшего регулирования и применении\n"
        "КонсультантПлюс\n"
        "надежная правовая поддержка www.consultant.ru Страница  2 из 271\n"
        "Документ предоставлен КонсультантПлюс\n"
        "Дата сохранения: 12.01.2026\n"
        '"Трудовой кодекс Российской Федерации" от 30.12.2001 N\n'
        "197-ФЗ\n"
        "(ред. от 28.12.2025)\n"
        "трудового законодательства в предусмотренных случаях;"
    )

    def test_removes_all_furniture(self):
        cleaned = rag.strip_boilerplate(self.RAW)
        for marker in ("consultant.ru", "Дата сохранения", "Документ предоставлен",
                       "надежная правовая поддержка", "Страница ", "197-ФЗ"):
            self.assertNotIn(marker, cleaned, f"не вычищено: {marker}")

    def test_rejoins_sentence_across_page_break(self):
        cleaned = rag.strip_boilerplate(self.RAW)
        self.assertIn("применении\nтрудового законодательства", cleaned)

    def test_keeps_real_amendment_notes(self):
        # настоящую сноску правки «(в ред. Федерального закона ...)» трогать нельзя,
        # удаляется только колонтитульная короткая «(ред. от ДД.ММ.ГГГГ)»
        text = "Статья 5. Норма.\n(в ред. Федерального закона от 30.06.2006 N 90-ФЗ)\nТекст."
        cleaned = rag.strip_boilerplate(text)
        self.assertIn("(в ред. Федерального закона от 30.06.2006 N 90-ФЗ)", cleaned)

    def test_keeps_clean_text(self):
        cleaned = rag.strip_boilerplate("строка один\nстрока два")
        self.assertIn("строка один", cleaned)
        self.assertIn("строка два", cleaned)


class TestLooksBinary(unittest.TestCase):
    def test_detects_raw_pdf(self):
        self.assertTrue(rag.looks_binary("%PDF-1.3\n1 0 obj\n<</Type /Catalog>>\nendobj"))

    def test_detects_control_char_garbage(self):
        self.assertTrue(rag.looks_binary("\x00\x01\x02\x03\x04" * 100 + "текст"))

    def test_normal_text_is_not_binary(self):
        self.assertFalse(rag.looks_binary("Статья 1. Обычный текст закона с переносами.\nВторая строка."))


class TestCosine(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(rag.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(rag.cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector_no_div_by_zero(self):
        self.assertEqual(rag.cosine([0.0, 0.0], [1.0, 1.0]), 0.0)


class TestExtractText(unittest.TestCase):
    def test_reads_txt_utf8(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.txt"
            f.write_text("привет мир", encoding="utf-8")
            self.assertEqual(rag.extract_text(f), "привет мир")


# ── end-to-end пайплайн + качество поиска (фейковый эмбеддер, изолированный индекс) ──
class TestRagIndexPipeline(unittest.TestCase):
    LEGAL = (
        "Статья 1. Цели трудового законодательства. Установление государственных гарантий.\n"
        "Статья 2. Основные принципы. Свобода труда, запрещение дискриминации.\n"
        "Статья 114. Ежегодный оплачиваемый отпуск. Работникам предоставляется ежегодный "
        "оплачиваемый отпуск с сохранением места работы и среднего заработка."
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # изолируем индекс от реального data/rag_index.json
        self._orig_data, self._orig_index = rag.DATA_DIR, rag.INDEX_PATH
        rag.DATA_DIR = Path(self._tmp.name)
        rag.INDEX_PATH = rag.DATA_DIR / "rag_index.json"
        self.src = Path(self._tmp.name) / "tk.txt"
        self.src.write_text(self.LEGAL, encoding="utf-8")

    def tearDown(self):
        rag.DATA_DIR, rag.INDEX_PATH = self._orig_data, self._orig_index
        self._tmp.cleanup()

    def _index(self):
        return rag.RagIndex(FakeClient(), "fake-embed")

    def test_add_path_indexes_articles(self):
        idx = self._index()
        files, chunks = idx.add_path(str(self.src))
        self.assertEqual(files, 1)
        self.assertEqual(chunks, 3)  # три статьи -> три чанка
        self.assertTrue(rag.INDEX_PATH.exists())

    def test_search_retrieves_right_article(self):
        """Проверка качества: запрос про отпуск достаёт именно статью 114."""
        idx = self._index()
        idx.add_path(str(self.src))
        hits = idx.search("ежегодный оплачиваемый отпуск работнику", top_k=1, min_score=0.0)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0]["text"].startswith("Статья 114."))

    def test_search_hit_has_id_and_score(self):
        # для структурированного вывода (источники + цитаты) хиту нужны chunk_id и score
        idx = self._index()
        idx.add_path(str(self.src))
        hit = idx.search("отпуск", top_k=1, min_score=0.0)[0]
        self.assertIn("chunk_id", hit)
        self.assertIsInstance(hit["chunk_id"], int)
        self.assertTrue(0.0 <= hit["score"] <= 1.0)

    def test_search_empty_index(self):
        self.assertEqual(self._index().search("что угодно"), [])

    def test_min_score_filters_out_irrelevant(self):
        idx = self._index()
        idx.add_path(str(self.src))
        # запрос без общих слов со статьями -> высокий порог отсекает всё
        hits = idx.search("квантовая хромодинамика адронов", top_k=3, min_score=0.5)
        self.assertEqual(hits, [])

    def test_dedup_keeps_one_chunk_per_article(self):
        # два чанка одной статьи (дубли по норме) + другая статья
        idx = self._index()
        texts = ["Статья 5. отпуск отпуск первая часть",
                 "Статья 5. отпуск вторая часть",
                 "Статья 6. отпуск иная норма"]
        idx.entries = [{"source": "x.txt", "text": t, "vector": _fake_vector(t)} for t in texts]
        plain = [rag.section_of(h["text"]) for h in idx.search("отпуск", top_k=3, min_score=0.0)]
        self.assertNotEqual(len(plain), len(set(plain)))  # без дедупа ст.5 занимает 2 слота
        ded = [rag.section_of(h["text"]) for h in idx.search("отпуск", top_k=3, min_score=0.0, dedup=True)]
        self.assertEqual(len(ded), len(set(ded)))         # все нормы разные
        self.assertIn("Статья 6.", ded)                   # дедуп освободил слот другой статье

    def test_reindex_replaces_old_chunks(self):
        idx = self._index()
        idx.add_path(str(self.src))
        self.src.write_text("Статья 1. Только одна статья теперь.", encoding="utf-8")
        idx.add_path(str(self.src))
        self.assertEqual(len(idx.entries), 1)  # старые 3 чанка заменены

    def test_sources_counts_chunks_per_file(self):
        idx = self._index()
        idx.add_path(str(self.src))
        self.assertEqual(idx.sources(), {str(self.src): 3})

    def test_rejects_binary_file_indexed_as_text(self):
        # ровно тот провал, что был в проде: сырой PDF попал бы в индекс байтами
        self.src.write_text("%PDF-1.3\n1 0 obj\n<</Type /Catalog>>\nendobj", encoding="utf-8")
        idx = self._index()
        with self.assertRaises(RuntimeError):
            idx.add_path(str(self.src))
        self.assertEqual(idx.entries, [])  # ничего не проиндексировано


# ── улучшение ретривала: rewriter, фильтр релевантности, reranker ────────────
class TestRelevanceFilter(unittest.TestCase):
    def test_keeps_top_cluster_drops_weak(self):
        hits = [{"text": "a", "score": 0.80}, {"text": "b", "score": 0.75}, {"text": "c", "score": 0.50}]
        out = rag.relevance_filter(hits, gap=0.12)
        self.assertEqual([h["text"] for h in out], ["a", "b"])  # c слишком слаб (0.50 < 0.68)

    def test_empty(self):
        self.assertEqual(rag.relevance_filter([]), [])


class TestRewriteQuery(unittest.TestCase):
    HIST = [{"role": "user", "content": "Как реагировать на травмы?"},
            {"role": "assistant", "content": "По ФЗ-125 и ТК РФ..."}]

    def test_uses_llm_rewrite(self):
        c = FakeChatClient("Какой закон регулирует расследование производственных травм")
        out = rag.rewrite_query(c, "m", self.HIST, "а по какому закону?")
        self.assertIn("закон", out.lower())

    def test_no_history_returns_original(self):
        self.assertEqual(rag.rewrite_query(FakeChatClient("нечто"), "m", [], "вопрос"), "вопрос")

    def test_empty_llm_falls_back_to_question(self):
        self.assertEqual(rag.rewrite_query(FakeChatClient(""), "m", self.HIST, "вопрос"), "вопрос")


class TestRerank(unittest.TestCase):
    HITS = [{"text": "первый", "score": 0.7},
            {"text": "второй", "score": 0.6},
            {"text": "третий", "score": 0.5}]

    def test_reorders_by_llm(self):
        out = rag.rerank(FakeChatClient("2, 1"), "m", "q", self.HITS, top_k=3)
        self.assertEqual([h["text"] for h in out], ["второй", "первый"])

    def test_none_relevant_returns_empty(self):
        self.assertEqual(rag.rerank(FakeChatClient("НЕТ"), "m", "q", self.HITS), [])

    def test_unparseable_falls_back_to_topk(self):
        out = rag.rerank(FakeChatClient("no numbers here"), "m", "q", self.HITS, top_k=2)
        self.assertEqual([h["text"] for h in out], ["первый", "второй"])

    def test_strips_think_block(self):
        out = rag.rerank(FakeChatClient("<think>долго думаю</think>1"), "m", "q", self.HITS)
        self.assertEqual([h["text"] for h in out], ["первый"])

    def test_empty_hits(self):
        self.assertEqual(rag.rerank(FakeChatClient("1"), "m", "q", []), [])


# ── опциональная проверка на реальном PDF (если задан RAG_TEST_PDF) ───────────
@unittest.skipUnless(os.environ.get("RAG_TEST_PDF"), "переменная RAG_TEST_PDF не задана")
class TestRealPdf(unittest.TestCase):
    """Прогон на настоящем экспорте КонсультантПлюс: RAG_TEST_PDF=<путь> python -m unittest ..."""

    def test_clean_and_structured(self):
        text = rag.extract_text(Path(os.environ["RAG_TEST_PDF"]))
        self.assertNotIn("Документ предоставлен КонсультантПлюс", text)
        self.assertNotIn("www.consultant.ru", text)
        chunks = rag.structured_chunks(text)
        self.assertIsNotNone(chunks, "статьи не распознаны — structured вернул None")
        self.assertGreater(len(chunks), 100)
        self.assertTrue(any(c.startswith("Статья 1.") for c in chunks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
