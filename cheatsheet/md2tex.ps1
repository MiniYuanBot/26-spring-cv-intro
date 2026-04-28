# md2tex.ps1

$inputDir  = "./mid-term/markdown-notes"
$outputDir = "./mid-term/source"
$assetsDir = Join-Path $inputDir "assets"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# 1. SVG -> PDF (LaTeX 不能直接包含 SVG)
if (Test-Path $assetsDir) {
    Get-ChildItem -Path $assetsDir -Filter "*.svg" | ForEach-Object {
        $pdfPath = $_.FullName -replace '\.svg$', '.pdf'
        if (-not (Test-Path $pdfPath)) {
            if (Get-Command inkscape -ErrorAction SilentlyContinue) {
                inkscape $_.FullName --export-filename=$pdfPath 2>$null
                Write-Host "SVG->PDF: $($_.Name)"
            } elseif (Get-Command rsvg-convert -ErrorAction SilentlyContinue) {
                rsvg-convert -f pdf -o $pdfPath $_.FullName
                Write-Host "SVG->PDF: $($_.Name)"
            } else {
                Write-Warning "No SVG converter found. Install Inkscape or librsvg."
            }
        }
    }
}

# 2. Markdown -> TeX
Get-ChildItem -Path $inputDir -Filter "*.md" | ForEach-Object {
    $texName    = $_.BaseName + ".tex"
    $outputPath = Join-Path $outputDir $texName

    pandoc $_.FullName -o $outputPath `
        --standalone `
        --from markdown-simple_tables-multiline_tables-pipe_tables `
        --resource-path=$inputDir `
        --variable=graphics `
        --variable=graphics-extension=.pdf

    Write-Host "Converted: $($_.Name) -> $texName"
}

Write-Host "`nDone. Output in: $(Resolve-Path $outputDir)"