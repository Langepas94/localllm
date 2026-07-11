param(
    [string]$HostAlias = "tg-agent-vps",
    [string]$RemoteDir = "/opt/localllm-web",
    [string]$ServiceName = "localllm-web.service"
)

# Деплой веб-агента localllm на VPS ровно из git: на сервер попадают ТОЛЬКО отслеживаемые
# git-файлы (git ls-files). Удалённая папка стирается (кроме .venv и data — индекс) и
# распаковывается заново. Гарантия: сервер == git, локально и на сервере одна программа.

$ErrorActionPreference = "Stop"

function Run($File, [string[]]$ArgsList) {
    & $File @ArgsList
    if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$tmpId = [System.Guid]::NewGuid().ToString("N")
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("localllm-web-" + $tmpId + ".tar")
$fileList = Join-Path ([System.IO.Path]::GetTempPath()) ("localllm-web-files-" + $tmpId + ".txt")
$scriptFile = Join-Path ([System.IO.Path]::GetTempPath()) ("localllm-web-remote-" + $tmpId + ".sh")
$remoteTmp = "/tmp/localllm-web-deploy.tar"
$remoteScriptPath = "/tmp/localllm-web-remote.sh"

try {
    Push-Location $repoRoot

    Write-Host "==> package localllm from git-tracked files"
    $files = git ls-files
    if ($LASTEXITCODE -ne 0) { throw "git ls-files failed with exit code $LASTEXITCODE" }
    if (-not $files) { throw "no git-tracked files found" }
    [System.IO.File]::WriteAllLines($fileList, [string[]]$files, [System.Text.UTF8Encoding]::new($false))
    Run "tar" @("-cf", $tmp, "-C", $repoRoot, "-T", $fileList)

    Write-Host "==> upload package to $HostAlias"
    Run "scp" @($tmp, "${HostAlias}:${remoteTmp}")

    $remoteScript = @"
set -euo pipefail
REMOTE_DIR="$RemoteDir"
REMOTE_TMP="$remoteTmp"
SERVICE_NAME="$ServiceName"

# 1. Разложить ровно git-файлы (сохраняем venv и собранный индекс в data/)
mkdir -p "`$REMOTE_DIR"
find "`$REMOTE_DIR" -mindepth 1 -maxdepth 1 ! -name .venv ! -name data -exec rm -rf -- {} +
tar -xf "`$REMOTE_TMP" -C "`$REMOTE_DIR"
rm -f "`$REMOTE_TMP"

cd "`$REMOTE_DIR"

# 2. venv из requirements.txt (openai, httpx, prompt_toolkit, pypdf)
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements.txt >/dev/null

# 3. Индекс ТК РФ — собрать вашим rag.py, только если его ещё нет (артефакт, не в git)
if [ ! -f data/rag_index.json ]; then
  echo "==> build RAG index from corpus/ (rag.py, bge-m3 via Ollama)"
  .venv/bin/python scripts/build_index.py --url http://127.0.0.1:11434/v1 --embed-model bge-m3
else
  echo "==> index already present, skip build"
fi

# 4. systemd-юнит из git
cp systemd/localllm-web.service /etc/systemd/system/localllm-web.service
systemctl daemon-reload
systemctl enable "`$SERVICE_NAME" >/dev/null 2>&1 || true

# 5. nginx-конфиги из git (rate-limit + сайт как единственный default_server)
cp nginx/ratelimit.conf /etc/nginx/conf.d/localllm-ratelimit.conf
cp nginx/localllm-web.conf /etc/nginx/sites-available/localllm-web
ln -sf /etc/nginx/sites-available/localllm-web /etc/nginx/sites-enabled/localllm-web
rm -f /etc/nginx/sites-enabled/tg-agent /etc/nginx/sites-enabled/labor-law-agent
nginx -t
systemctl reload nginx

# 6. перезапуск сервиса
systemctl restart "`$SERVICE_NAME"

# 7. smoke-тест health (ждём прогрев)
for i in `$(seq 1 20); do
  if curl -sf -m 10 http://127.0.0.1:8000/api/health >/dev/null; then
    echo "==> health OK"; curl -s http://127.0.0.1:8000/api/health; echo; exit 0
  fi
  sleep 3
done
echo "!! health-тест не прошёл" >&2
systemctl status "`$SERVICE_NAME" --no-pager | tail -20 >&2
exit 1
"@

    Write-Host "==> deploy on $HostAlias"
    # Remote-скрипт файлом без BOM + LF, запуск по scp+ssh (пайп в ssh в PowerShell 5.1
    # добавляет UTF-8 BOM -> первая строка "﻿set" и strict-mode не включается).
    [System.IO.File]::WriteAllText($scriptFile, ($remoteScript -replace "`r`n", "`n"), [System.Text.UTF8Encoding]::new($false))
    Run "scp" @($scriptFile, "${HostAlias}:${remoteScriptPath}")
    & ssh $HostAlias "bash $remoteScriptPath; rc=`$?; rm -f $remoteScriptPath; exit `$rc"
    if ($LASTEXITCODE -ne 0) { throw "remote deploy failed with exit code $LASTEXITCODE" }

    Write-Host "==> done: http://5.129.234.9/"
}
finally {
    Pop-Location
    Remove-Item -Force -ErrorAction SilentlyContinue $tmp, $fileList, $scriptFile
}
