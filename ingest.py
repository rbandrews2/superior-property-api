# ingest.py
import os, uuid, asyncpg, glob
from main import embed, pg  # reuse

async def ingest_dir(path: str, source: str):
    files = glob.glob(f"{path}/**/*.txt", recursive=True)
    texts = [open(f, "r", encoding="utf-8").read() for f in files]
    vecs = await embed(texts)
    conn = await pg()
    for f, t, v in zip(files, texts, vecs):
        await conn.execute(
          "INSERT INTO documents(id,address,title,content,source,embedding) VALUES($1,$2,$3,$4,$5,$6)",
          uuid.uuid4(), None, os.path.basename(f), t, source, v
        )
    await conn.close()
