param([switch]$Approve, [switch]$Phone, [string]$Signal = '', [double]$BudgetEur = 0)
# LIVE Trading 212 (real money). Dry-run by default; -Approve shows the exact order and sends ONE after you type "yes live".
# Refuses unless config\private_live.yaml opens the gate. Run this yourself in a terminal; agents cannot answer the prompt.
$ErrorActionPreference = 'Stop'
if (-not $Signal) { throw 'Pass -Signal data\ai_signals\<live signal>.json (its sleeve must be "live").' }
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Python 3.12 must be installed before running the bot.' }
foreach ($variableName in @('TRADING212_LIVE_API_KEY', 'TRADING212_LIVE_API_SECRET')) {
    $credentialValue = [Environment]::GetEnvironmentVariable($variableName, 'User')
    if (-not [string]::IsNullOrWhiteSpace($credentialValue)) { Set-Item -Path "Env:$variableName" -Value $credentialValue }
}
$credentialValue = $null
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$runnerArguments = @((Join-Path $PSScriptRoot 'live_trade.py'), '--signal', (Resolve-Path -LiteralPath $Signal).Path)
if ($Approve) { $runnerArguments += '--approve' }
if ($Phone) { $runnerArguments += '--phone' }
if ($BudgetEur -gt 0) { $runnerArguments += @('--budget-eur', $BudgetEur.ToString([Globalization.CultureInfo]::InvariantCulture)) }
Push-Location -LiteralPath $projectRoot
try { & $pythonPath @runnerArguments; $runnerExitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $runnerExitCode
