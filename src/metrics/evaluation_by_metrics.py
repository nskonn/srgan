import cv2

hr_img_1 = cv2.imread('./images/0051.png')
hr_img_2 = cv2.imread('./images/0067.png')
hr_img_3 = cv2.imread('./images/0068.png')
sr_img_1 = cv2.imread('./images/result_141.jpg')
sr_img_2 = cv2.imread('./images/result_150_67.jpg')
sr_img_3 = cv2.imread('./images/result_150_68.jpg')


psnr_3 = cv2.PSNR(hr_img_3, sr_img_3)
psnr_1 = cv2.PSNR(hr_img_1, sr_img_1)
psnr_2 = cv2.PSNR(hr_img_2, sr_img_2)

print(f"PSNR image_1: {psnr_1}")
print(f"PSNR image_2: {psnr_2}")
print(f"PSNR image_3: {psnr_3}")
