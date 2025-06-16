import sqlite3
import pandas as pd
import shutil
import os

def combine_optuna_dbs_to_new(merged_db_path, db_paths):
    """
    Merges multiple Optuna SQLite databases into a new database file.

    Parameters:
        merged_db_path (str): Path to the new merged database file (will be created or overwritten).
        db_paths (list of str): List of database paths to merge. The first DB is used as a base.

    Notes:
        - All databases must have the same study structure.
        - Ensures trial_id and related IDs are uniquely updated to prevent collisions.
    """
    if os.path.exists(merged_db_path):
        os.remove(merged_db_path)  # Clean up if already exists

    if not db_paths:
        raise ValueError("No database paths provided.")

    # Copy the first DB as the base
    shutil.copyfile(db_paths[0], merged_db_path)

    # Read table names from the merged DB (excluding metadata)
    with sqlite3.connect(merged_db_path) as con:
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        tables = [t[0] for t in tables]
    exclude_tables = ['studies', 'version_info', 'study_directions', 'alembic_version']
    tables = [t for t in tables if t not in exclude_tables]

    # Load base tables to track max IDs
    dfs_base = {}
    with sqlite3.connect(merged_db_path) as con:
        for table in tables:
            dfs_base[table] = pd.read_sql_query(f"SELECT * FROM {table}", con)

    table_ids = {
        'study_user_attributes': ['study_user_attribute_id'],
        'study_system_attributes': ['study_system_attribute_id'],
        'trials': ['trial_id', 'number'],
        'trial_user_attributes': ['trial_user_attribute_id', 'trial_id'],
        'trial_system_attributes': ['trial_system_attribute_id', 'trial_id'],
        'trial_params': ['param_id', 'trial_id'],
        'trial_values': ['trial_value_id', 'trial_id'],
        'trial_intermediate_values': ['trial_intermediate_value_id', 'trial_id'],
        'trial_heartbeats': ['trial_heartbeat_id', 'trial_id'],
    }


    # Process and merge each additional DB
    for db_path in db_paths[1:]:
        # Compute max IDs for initial state
        max_ids = {}
        for table in tables:
            for col in table_ids.get(table, []):
                if col in dfs_base[table].columns and not dfs_base[table].empty:
                    max_ids[col] = dfs_base[table][col].max()
                else:
                    max_ids[col] = 0
        dfs2 = {}
        with sqlite3.connect(db_path) as con:
            for table in tables:
                dfs2[table] = pd.read_sql_query(f"SELECT * FROM {table}", con)

        update_ids = {'study_user_attributes' : ['study_user_attribute_id'],
                'study_system_attributes' : ['study_system_attribute_id'],
                'trials' : ['trial_id', 'number'],
                'trial_user_attributes' : ['trial_user_attribute_id', 'trial_id'],
                'trial_system_attributes' : ['trial_system_attribute_id', 'trial_id'],
                'trial_params' : ['param_id', 'trial_id'],
                'trial_values' : ['trial_value_id', 'trial_id'],
                'trial_intermediate_values' : ['trial_intermediate_value_id', 'trial_id'],
                'trial_heartbeats' : ['trial_heartbeat_id', 'trial_id'],
        }

        # update the ids of the second db
        for table in tables:
            if len(dfs2[table]) > 0:
                for id in update_ids[table]:
                    dfs2[table][id] = dfs2[table][id] + max_ids[id]
        print(dfs2)
        # add the second db to the first db
        with sqlite3.connect(merged_db_path) as con1:
            for table in tables:
                dfs2[table].to_sql(table, con1, if_exists='append', index=False)



if __name__ == "__main__":
    import glob

    files = glob.glob(
        "C:/Users/edmun/OneDrive/Desktop/Quantitative Trading Strategies/Project/qts/output/polygon/*.db"
    )
    print(files)
    combine_optuna_dbs_to_new(
        "C:/Users/edmun/OneDrive/Desktop/Quantitative Trading Strategies/Project/qts/output/polygon/merged.db",
        files,
    )
