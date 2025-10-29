# ML Resume Matcher - Deployment Guide

## Deployment Options

### 1. Heroku Deployment (Recommended for beginners)

#### Prerequisites
- Heroku CLI installed
- Git repository

#### Steps
```bash
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-resume-matcher-app

# Set environment variables
heroku config:set FLASK_ENV=production

# Deploy
git add .
git commit -m "Prepare for deployment"
git push heroku main

# Open your app
heroku open
```

#### Important Notes for Heroku
- The app uses ML models which require more memory
- Consider using Heroku's Performance dynos for better performance
- First load might be slow due to model loading

### 2. Railway Deployment

#### Steps
1. Connect your GitHub repository to Railway
2. Railway will auto-detect the Python app
3. Set environment variable: `FLASK_ENV=production`
4. Deploy automatically

### 3. Render Deployment

#### Steps
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
4. Set environment variable: `FLASK_ENV=production`

### 4. Docker Deployment

#### Local Docker
```bash
# Build the image
docker build -t resume-matcher .

# Run the container
docker run -p 5009:5009 resume-matcher
```

#### Docker Compose
```bash
# Create docker-compose.yml and run
docker-compose up
```

### 5. VPS/Cloud Server Deployment

#### Using Ubuntu/Debian server
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv nginx -y

# Clone your repository
git clone <your-repo-url>
cd resumeparser

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install and configure Gunicorn
pip install gunicorn

# Create systemd service
sudo nano /etc/systemd/system/resume-matcher.service
```

#### Systemd Service File
```ini
[Unit]
Description=Resume Matcher Flask App
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/your/app
Environment="PATH=/path/to/your/app/venv/bin"
ExecStart=/path/to/your/app/venv/bin/gunicorn --workers 3 --bind unix:resume-matcher.sock -m 007 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/path/to/your/app/resume-matcher.sock;
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
    }

    client_max_body_size 10M;
}
```

## Environment Variables

- `FLASK_ENV`: Set to `production` for production deployment
- `PORT`: Port number (automatically set by most platforms)

## Performance Considerations

1. **Memory Usage**: The sentence-transformer model requires ~500MB RAM
2. **Cold Starts**: First request might be slow due to model loading
3. **File Uploads**: Limited to 16MB by default
4. **Timeout**: Set to 120 seconds for model processing

## Monitoring and Logs

- Check application logs for errors
- Monitor memory usage (especially important for ML models)
- Set up health checks if available on your platform

## Security Notes

- The app creates temporary files for PDF processing
- Files are automatically cleaned up after processing
- Consider adding rate limiting for production use
- Use HTTPS in production

## Troubleshooting

### Common Issues
1. **Memory errors**: Increase dyno/instance memory
2. **Timeout errors**: Increase timeout settings
3. **Model loading errors**: Check Python version compatibility
4. **File upload errors**: Check file size limits

### Debug Mode
Set `FLASK_ENV=development` to enable debug mode (not recommended for production)