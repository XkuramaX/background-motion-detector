from datetime import datetime
import pandas as pd
import cv2,time
df=pd.DataFrame(columns=["Entry","Exit","Difference"])
first_frame=None
status_list=[None,None]
times=[]
file=0
flag=0
time=50
c=0
video=cv2.VideoCapture(0)
while True:
	check,frame=video.read()
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray=cv2.GaussianBlur(gray,(21,21),0)
	if first_frame is None:
		first_frame=gray
		continue
	delta_frame=cv2.absdiff(first_frame,gray)
	thresh_frame=cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
	status=0
	thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)
	(cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for contours in cnts:
		if cv2.contourArea(contours)<5000:
			continue
		status=1
		(x,y,w,h)=cv2.boundingRect(contours)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

	status_list.append(status)
	if status==1 and flag==0:
		c+=1
		if c==time:
			flag=1
			cv2.imwrite(str(file)+".jpg",frame)
			c=0
			print(str(file))
	if status==0 and flag==1:
		flag=0
		file+=1

	if status_list[-1]==1 and status_list[-2]==0:
		times.append(datetime.now())
	if status_list[-1]==0 and status_list[-2]==1:
		times.append(datetime.now())
	cpy_frame=frame
	cv2.imshow("Video Window",cv2.resize(cpy_frame,(cpy_frame.shape[1]//3,cpy_frame.shape[0]//3)))

	key=cv2.waitKey(1)
	if key==ord('q'):
		if status==1:
			times.append(datetime.now())
		break
print(times)

for i in range(0,len(times),2):
	df.append({"Entry time":times[i],"Exit time":times[i+1],"Difference":times[i+1]-times[i]},ignore_index=True)
df.to_csv("EntryExit.csv")


video.release()
cv2.destroyAllWindows