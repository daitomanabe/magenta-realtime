// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MRT2 - Sequencer standalone scaffold.

#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>

@interface SequencerAppDelegate : NSObject <NSApplicationDelegate, WKScriptMessageHandler, WKNavigationDelegate>
@end

@implementation SequencerAppDelegate {
    NSWindow* _window;
    WKWebView* _webView;
    BOOL _isPlaying;
    NSString* _patternName;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    (void)notification;
    _isPlaying = NO;
    _patternName = @"8 Bar Tension Chords";

    [self buildMenu];
    [self buildWindow];
    [self loadSequencerUI];
}

- (void)applicationWillTerminate:(NSNotification*)notification {
    (void)notification;
    [_webView.configuration.userContentController removeScriptMessageHandlerForName:@"sequencerHost"];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
    (void)sender;
    return YES;
}

- (void)buildMenu {
    NSMenu* menuBar = [[NSMenu alloc] init];

    NSMenuItem* appMenuItem = [[NSMenuItem alloc] init];
    NSMenu* appMenu = [[NSMenu alloc] initWithTitle:@"MRT2 - Sequencer"];
    NSString* quitTitle = @"Quit MRT2 - Sequencer";
    [appMenu addItemWithTitle:quitTitle action:@selector(terminate:) keyEquivalent:@"q"];
    appMenuItem.submenu = appMenu;
    [menuBar addItem:appMenuItem];

    NSMenuItem* transportMenuItem = [[NSMenuItem alloc] init];
    NSMenu* transportMenu = [[NSMenu alloc] initWithTitle:@"Transport"];
    [transportMenu addItemWithTitle:@"Play / Stop" action:@selector(togglePlay:) keyEquivalent:@" "];
    [transportMenu addItemWithTitle:@"Panic" action:@selector(panic:) keyEquivalent:@"."];
    transportMenuItem.submenu = transportMenu;
    [menuBar addItem:transportMenuItem];

    [NSApp setMainMenu:menuBar];
}

- (void)buildWindow {
    NSRect frame = NSMakeRect(0, 0, 1280, 820);
    _window = [[NSWindow alloc] initWithContentRect:frame
                                         styleMask:NSWindowStyleMaskTitled |
                                                   NSWindowStyleMaskClosable |
                                                   NSWindowStyleMaskResizable |
                                                   NSWindowStyleMaskMiniaturizable
                                           backing:NSBackingStoreBuffered
                                             defer:NO];
    _window.title = @"MRT2 - Sequencer";
    _window.minSize = NSMakeSize(980, 680);

    WKWebViewConfiguration* config = [[WKWebViewConfiguration alloc] init];
    WKWebpagePreferences* pagePreferences = [[WKWebpagePreferences alloc] init];
    pagePreferences.allowsContentJavaScript = YES;
    config.defaultWebpagePreferences = pagePreferences;
    [config.userContentController addScriptMessageHandler:self name:@"sequencerHost"];

    _webView = [[WKWebView alloc] initWithFrame:_window.contentView.bounds configuration:config];
    _webView.navigationDelegate = self;
    _webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [_window.contentView addSubview:_webView];

    [_window center];
    [_window makeKeyAndOrderFront:nil];
}

- (void)loadSequencerUI {
    NSURL* uiURL = [[NSBundle mainBundle] URLForResource:@"index"
                                           withExtension:@"html"
                                            subdirectory:@"sequencer_ui"];
    if (uiURL) {
        [_webView loadFileURL:uiURL allowingReadAccessToURL:uiURL.URLByDeletingLastPathComponent];
        return;
    }

    NSString* fallback = @"<!doctype html><meta charset='utf-8'><body style='font:13px -apple-system; padding:24px'>Sequencer UI not built.</body>";
    [_webView loadHTMLString:fallback baseURL:nil];
}

- (void)webView:(WKWebView*)webView didFinishNavigation:(WKNavigation*)navigation {
    (void)webView;
    (void)navigation;
    [self sendStateUpdate:@{
        @"isPlaying": @(_isPlaying),
        @"modelName": @"No model loaded",
        @"patternName": _patternName,
        @"nativeReady": @YES,
    }];
}

- (void)sendStateUpdate:(NSDictionary*)state {
    if (!_webView || state.count == 0) return;

    NSError* error = nil;
    NSData* data = [NSJSONSerialization dataWithJSONObject:state options:0 error:&error];
    if (!data || error) {
        NSLog(@"Sequencer: failed to serialize state update: %@", error);
        return;
    }

    NSString* json = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    NSString* js = [NSString stringWithFormat:@"window.updateState && window.updateState(%@);", json];
    dispatch_async(dispatch_get_main_queue(), ^{
        [self->_webView evaluateJavaScript:js completionHandler:nil];
    });
}

- (void)togglePlay:(id)sender {
    (void)sender;
    _isPlaying = !_isPlaying;
    [self sendStateUpdate:@{@"isPlaying": @(_isPlaying)}];
}

- (void)panic:(id)sender {
    (void)sender;
    _isPlaying = NO;
    [self sendStateUpdate:@{@"isPlaying": @NO, @"panicCount": @([[NSDate date] timeIntervalSince1970])}];
}

- (void)userContentController:(WKUserContentController*)userContentController didReceiveScriptMessage:(WKScriptMessage*)message {
    (void)userContentController;
    if (![message.name isEqualToString:@"sequencerHost"] || ![message.body isKindOfClass:[NSDictionary class]]) {
        return;
    }

    NSDictionary* body = (NSDictionary*)message.body;
    NSString* type = body[@"type"];
    if (![type isKindOfClass:[NSString class]]) return;

    if ([type isEqualToString:@"uiReady"]) {
        [self webView:_webView didFinishNavigation:nil];
    } else if ([type isEqualToString:@"sequencerTransport"]) {
        NSNumber* playing = body[@"playing"];
        if ([playing isKindOfClass:[NSNumber class]]) {
            _isPlaying = playing.boolValue;
            [self sendStateUpdate:@{@"isPlaying": @(_isPlaying)}];
        }
    } else if ([type isEqualToString:@"sequencerPattern"]) {
        NSDictionary* pattern = body[@"pattern"];
        NSString* name = pattern[@"name"];
        if ([name isKindOfClass:[NSString class]]) {
            _patternName = [name copy];
        }
        [self sendStateUpdate:@{@"patternName": _patternName, @"patternSynced": @YES}];
    } else if ([type isEqualToString:@"sequencerPanic"]) {
        [self panic:nil];
    } else if ([type isEqualToString:@"sequencerRender"]) {
        [self sendStateUpdate:@{@"renderStatus": @"queued"}];
    } else if ([type isEqualToString:@"log"]) {
        NSString* value = body[@"value"];
        if ([value isKindOfClass:[NSString class]]) {
            NSLog(@"Sequencer UI: %@", value);
        }
    }
}

@end

int main(int argc, const char* argv[]) {
    (void)argc;
    (void)argv;
    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        SequencerAppDelegate* delegate = [[SequencerAppDelegate alloc] init];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
