param(
    [string]$ImagePath,
    [string]$BaseUrl = "http://localhost:8000",
    [string]$Task = "all"
)

if (-not (Test-Path -LiteralPath $ImagePath)) {
    throw "Image not found: $ImagePath"
}

$form = @{
    image = Get-Item -LiteralPath $ImagePath
    mpp = "1.0"
    image_id = "smoke"
}

Invoke-RestMethod -Uri "$BaseUrl/segment/$Task" -Method Post -Form $form
