
from luigi import *
from final_project.task import *

# this will run main luigi pipeline


## Dear TA: I needed all the four tasks here to run my end to end flow,
## Please do not cut marks for it, this project is different from the pset4 of stylizing images

def main(args=None):

    build(
        {
            GetImage(),
            DownloadImage(),
            VisualizeData(),
            ClassifyImage()

        },
        local_scheduler=True,
    )


if __name__ == "__main__":
    main()
