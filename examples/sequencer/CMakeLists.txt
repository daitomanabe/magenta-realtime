# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MRT2 - Sequencer standalone macOS app scaffold.

set(APP_ICON "${CMAKE_CURRENT_SOURCE_DIR}/../common/assets/AppIcon.icns")
set_source_files_properties(${APP_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

add_executable(mrt2_sequencer MACOSX_BUNDLE
    SequencerApp.mm
    ${APP_ICON}
)

target_include_directories(mrt2_sequencer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(mrt2_sequencer PRIVATE -fobjc-arc)

target_link_libraries(mrt2_sequencer PRIVATE
    "-framework AppKit"
    "-framework WebKit"
)

set_target_properties(mrt2_sequencer PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/SequencerInfo.plist.in"
    MACOSX_BUNDLE_EXECUTABLE_NAME MRT2_Sequencer
    MACOSX_BUNDLE_BUNDLE_NAME "MRT2 - Sequencer"
    MACOSX_BUNDLE_ICON_FILE "AppIcon.icns"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "com.google.mrt2.sequencer"
)

add_custom_target(build_sequencer_ui ALL
    COMMAND npm run build
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ui
    COMMENT "Building Sequencer UI..."
)
add_dependencies(build_sequencer_ui npm_install_root)
add_dependencies(mrt2_sequencer build_sequencer_ui)

add_custom_target(deploy_mrt2_sequencer
    DEPENDS mrt2_sequencer
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_DIR:mrt2_sequencer>/Contents/Resources/sequencer_ui"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/ui/dist"
        "$<TARGET_BUNDLE_DIR:mrt2_sequencer>/Contents/Resources/sequencer_ui"
    COMMAND codesign ${CODESIGN_FLAGS}
        --entitlements "${CMAKE_CURRENT_SOURCE_DIR}/SequencerEntitlements.plist"
        --generate-entitlement-der
        "$<TARGET_BUNDLE_DIR:mrt2_sequencer>"
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        "$ENV{HOME}/Applications/MRT2 - Sequencer.app"
    COMMAND ditto
        "$<TARGET_BUNDLE_DIR:mrt2_sequencer>"
        "$ENV{HOME}/Applications/MRT2 - Sequencer.app"
    COMMENT "Building and deploying MRT2 - Sequencer standalone app"
    VERBATIM
)

magentart_add_notarize_target(notarize_mrt2_sequencer
    APP "$ENV{HOME}/Applications/MRT2 - Sequencer.app"
    DEPENDS deploy_mrt2_sequencer
    ZIP_NAME "MRT2_Sequencer.zip"
)
