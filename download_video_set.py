import os
import requests
if __name__ == "__main__":

    links = [
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_01-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_01-10.mp4'
        },
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_02-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_02-10.mp4'
        },
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_03-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_03-10.mp4'
        },
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_04-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_04-10.mp4'
        },
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_05-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_05-10.mp4'
        },
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_06-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_06-10.mp4'
        },
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_07-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_07-10.mp4'
        },
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_08-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_08-10.mp4'
        },
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_09-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_09-10.mp4'
        },
        {
            "name": "G-C2L2P-Jun25-C-Jacob_q2_10-10.mp4",
            "link": 'https://ece46medsrv.ece.unm.edu/COHORT_2\LEVEL_2\POLK\08_Polk_Jun25\08_Polk_Jun25_GroupC\Group_Interactions\Jacob/G-C2L2P-Jun25-C-Jacob_q2_10-10.mp4'
        },
        
    ]
    directory = "/media/twelvetb/testing/activity-detection-training-service/data/raw/video"
    for video_link in links:
        print(f"Downloading {video_link['name']}...")   
        r = requests.get(video_link["link"], stream=True)

        with open(os.path.join(directory, video_link["name"]), 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
            print(f"Download completed")
            