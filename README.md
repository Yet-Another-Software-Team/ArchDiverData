# ArchDiverData

ArchDiverData is a repository containing all data cleaning scripts and database images for interacting with data used in ArchDiver.

The original dataset comes from the QScored dataset (2022) by Sharma et al.
The data can be found on [zenodo](https://zenodo.org/records/7484812)

## Repository Structure

- `\output`: contains output files from scripts
- `\queries`: contains queries used to setup the database
- `\raw_data`: contains cleaned data with unverified links
- `\scripts`: contains data cleaning scripts

## Data preparation
To setup the raw data from zenodo follow the outlined steps below.
You will need at minimum 250 GB of storage.

1. Download data from [zenodo](https://zenodo.org/records/7484812)
    - It is highly recommended to download each database file individually.
    - The entire database is roughly 112 GB in size.

2. Extract all the database files and put them in a folder

3. Within the folder, Combine the database files
    ```
    # Mac/Linux
    cat qscored_dump_25Jan2021?? > qscored_dump_25Jan2021.tar
    ```

After combining the database, feel free to delete the database zip files.

## Running the Posgres database
1. Run the database and pgadmin for viewing the data.
    ```
    docker compose up -d
    ```

2. In a terminal, open the database image.
    ```
    docker exec -it postgres_9.6 psql -U postgres
    ```

3. In the postgres CLI interface, run the following commands
    ```
    # Create database
    CREATE DATABASE qscoreddb WITH TEMPLATE=template0 ENCODING='UTF-8' LC_COLLATE='american_usa' LC_CTYPE='american_usa';

    # Grant user permissions
    GRANT ALL PRIVILEGES ON DATABASE qscoreddb TO postgres;
    ```

4. To import the data run the following commands in a terminal
    ```
    docker exec -i postgres_9.6 pg_restore -v -u postgres -d qscoreddb < "{path to your database file from data prep}"
    ```
    Note: This may take up to 1 hour to complete, depending on your PC specifications.

## License

MIT License - see the [LICENSE](LICENSE) file for details.