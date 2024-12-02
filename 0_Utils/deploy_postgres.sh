#!/bin/bash

# Exit on error
set -e

# Function to print status messages
print_status() {
    echo "====================================="
    echo "$1"
    echo "====================================="
}

# Update package lists
print_status "Updating package lists..."
sudo apt-get update

# Install PostgreSQL
print_status "Installing PostgreSQL..."
sudo apt-get install -y postgresql postgresql-contrib

# Start PostgreSQL service
print_status "Starting PostgreSQL service..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create a new PostgreSQL user and database
DB_NAME="crypto_data"
DB_USER="lob_user"
DB_PASSWORD=${DB_PASSWORD:-default_password}

print_status "Configuring PostgreSQL user and database..."
sudo -u postgres psql <<EOF
CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
CREATE DATABASE $DB_NAME OWNER $DB_USER;
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
EOF

# Optional: Configure PostgreSQL for remote access
ENABLE_REMOTE_ACCESS=${1:-false} # Pass "true" as argument to enable remote access
if [ "$ENABLE_REMOTE_ACCESS" = "true" ]; then
    print_status "Enabling remote access to PostgreSQL..."
    sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/g" /etc/postgresql/*/main/postgresql.conf
    echo "host all all 0.0.0.0/0 md5" | sudo tee -a /etc/postgresql/*/main/pg_hba.conf
    sudo systemctl restart postgresql
fi

# Print success message
print_status "PostgreSQL installation and configuration complete."
echo "Database Name: $DB_NAME"
echo "Username: $DB_USER"
echo "Password: $DB_PASSWORD"
echo "Remote Access Enabled: $ENABLE_REMOTE_ACCESS"
