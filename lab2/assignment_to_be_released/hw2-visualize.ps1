# visualize.ps1

Write-Host "=== CIFAR-10 Visualization ===" -ForegroundColor Cyan
$env:PYTHONPATH = $PWD
$env:TF_ENABLE_ONEDNN_OPTS = "0"

Write-Host "`nPlease select an option:" -ForegroundColor Yellow
Write-Host "1. Visualize network structure (network.py)"
Write-Host "2. Visualize input data (dataset.py)"
Write-Host "3. Train network (train.py)"
Write-Host "4. Run all"
Write-Host "5. Start TensorBoard"
Write-Host "0. Exit"

$choice = Read-Host "`nEnter number [0-5]"

switch ($choice) {
    "1" {
        Write-Host "`n[1] Visualizing network structure..." -ForegroundColor Green
        Push-Location cifar-10
        python network.py
        Pop-Location
        Write-Host "Done!" -ForegroundColor Cyan
    }
    "2" {
        Write-Host "`n[2] Visualizing input data..." -ForegroundColor Green
        Push-Location cifar-10
        python dataset.py
        Pop-Location
        Write-Host "Done!" -ForegroundColor Cyan
    }
    "3" {
        $lr = Read-Host "Enter learning rate (default: 1e-4)"
        if ($lr -eq "") { $lr = "1e-4" }
        Write-Host "`n[3] Training with learning rate $lr..." -ForegroundColor Green
        Push-Location cifar-10
        python train.py -e test -l $lr
        Pop-Location
    }
    "4" {
        Write-Host "`n[All] Running sequentially..." -ForegroundColor Green
        Push-Location cifar-10
        Write-Host "`n>>> Visualizing network structure" -ForegroundColor Yellow
        python network.py
        Write-Host "`n>>> Visualizing input data" -ForegroundColor Yellow
        python dataset.py
        $lr = Read-Host "`nEnter learning rate (default: 1e-4)"
        if ($lr -eq "") { $lr = "1e-4" }
        Write-Host "`n>>> Training network" -ForegroundColor Yellow
        python train.py -e test -l $lr
        Pop-Location
    }
    "5" {
        Write-Host "`n[5] Starting TensorBoard..." -ForegroundColor Green
        Push-Location experiments
        try {
            tensorboard --logdir .
        } finally {
            Pop-Location
            Write-Host "Returned to original directory" -ForegroundColor Cyan
        }
    }
    "0" {
        Write-Host "Exiting." -ForegroundColor Gray
        exit
    }
    default {
        Write-Host "Invalid input. Please rerun the script." -ForegroundColor Red
    }
}
