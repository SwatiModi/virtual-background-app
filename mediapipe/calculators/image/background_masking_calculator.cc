// Takes input video and mask as input, applies a virtual background using alpha blending

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include <iostream>
#include <vector>


//calculator BackgroundMaskingCalculator
namespace mediapipe {

class BackgroundMaskingCalculator : public CalculatorBase {
public:

    BackgroundMaskingCalculator() = default;
    ~BackgroundMaskingCalculator() override = default;
    //BackgroundMaskingCalculator() : initialized_(false){}

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  };

  REGISTER_CALCULATOR(BackgroundMaskingCalculator);

::mediapipe::Status BackgroundMaskingCalculator::GetContract (CalculatorContract *cc){

    cc->Inputs().Tag("IMAGE_CPU").Set<ImageFrame>();
    cc->Inputs().Tag("MASK_CPU").Set<ImageFrame>();
    cc->Outputs().Tag("OUTPUT_VIDEO").Set<ImageFrame>();

    return ::mediapipe::OkStatus();

}

::mediapipe::Status BackgroundMaskingCalculator::Open(CalculatorContext* cc) {

    return ::mediapipe::OkStatus();
}

::mediapipe::Status BackgroundMaskingCalculator::Process(CalculatorContext* cc) {

    ///////////////////////ANDROID Asset reading method
    cv::Mat background;

    // mediapipe::StatusOr<std::string> status = ::mediapipe::PathToResourceAsFile("clouds2.png"); 
    // mediapipe::StatusOr<std::string> status = ::mediapipe::PathToResourceAsFile("stars.png"); 
    mediapipe::StatusOr<std::string> status = ::mediapipe::PathToResourceAsFile("buildings.jpg"); 
    

    if (status.ok()){
    background = cv::imread(status.ValueOrDie(), 1);
    }

    ///////////////////////////////// DESKTOP Asset reading method
    // cv::Mat background = cv::imread("mediapipe/calculators/image/testdata/clouds2.png");
    // cv::Mat background = cv::imread("mediapipe/calculators/image/testdata/stars.png");
    // cv::Mat background = cv::imread("mediapipe/calculators/image/testdata/buildings.jpg");

    std::cout << " Process() of BgMasking" << std::endl;

    const auto& input_img = cc->Inputs().Tag("IMAGE_CPU").Get<ImageFrame>();
    cv::Mat input_mat = formats::MatView(&input_img);

    const auto& mask_img = cc->Inputs().Tag("MASK_CPU").Get<ImageFrame>();
    cv::Mat mask_mat = formats::MatView(&mask_img);

    // std::cout << mask_mat << std::endl;

    cv::cvtColor(background, background, cv::COLOR_BGR2RGB);
    cv::cvtColor(background, background, cv::COLOR_RGB2RGBA);
    cv::resize(background, background, input_mat.size());       

    // create binary bg mask
    cv::Mat channels[4];
    cv::split(mask_mat, channels);

    cv::Mat bg_mask;
    cv::Mat fg_mask;

    cv::Mat in[] = {channels[3], channels[3], channels[3], channels[3]};
    cv::merge(in, 4, bg_mask);

    cv::bitwise_not(bg_mask, fg_mask); 
    bg_mask = bg_mask/255.0;
    fg_mask = fg_mask/255.0;

    int k= 9;
    cv::GaussianBlur(bg_mask, bg_mask, cv::Size(k,k), 7);
    cv::GaussianBlur(fg_mask, fg_mask, cv::Size(k,k), 7);

    //create bg image = image*bgmask
    cv::Mat bg_image;
    cv::multiply(background,bg_mask, bg_image);

    // //create fg image = image*fgmask
    cv::Mat fg_image;
    cv::multiply(input_mat,fg_mask, fg_image);

    cv::Mat output_image;
    cv::add(fg_image, bg_image, output_image);

    std::unique_ptr<ImageFrame> output_frame(
        new ImageFrame(input_img.Format(), input_img.Width(), input_img.Height()));

    cv::Mat output_mat = formats::MatView(output_frame.get());

    output_image.copyTo(output_mat);
    cc->Outputs().Tag("OUTPUT_VIDEO").Add(output_frame.release(), cc->InputTimestamp());

    return ::mediapipe::OkStatus();

    }

// after defining calculator class, we need to register it with a macro invocation 
// REGISTER_CALCULATOR(calculator_class_name).
REGISTER_CALCULATOR(::mediapipe::BackgroundMaskingCalculator);
}
 //end namespace
