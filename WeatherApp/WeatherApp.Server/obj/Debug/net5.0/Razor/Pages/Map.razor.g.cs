#pragma checksum "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "29e3294b139b013178b330ffd38e5061313f8d36"
// <auto-generated/>
#pragma warning disable 1591
namespace WeatherApp.Server.Pages
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Components;
#nullable restore
#line 1 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using System.Net.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Forms;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Routing;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 7 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.AspNetCore.Components.Web.Virtualization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 8 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#nullable disable
#nullable restore
#line 9 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using WeatherApp.Server;

#line default
#line hidden
#nullable disable
#nullable restore
#line 10 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\_Imports.razor"
using WeatherApp.Server.Shared;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
using GoogleMapsComponents;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
using GoogleMapsComponents.Maps;

#line default
#line hidden
#nullable disable
    [Microsoft.AspNetCore.Components.RouteAttribute("/map")]
    public partial class Map : Microsoft.AspNetCore.Components.ComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
            __builder.OpenComponent<Microsoft.AspNetCore.Components.Authorization.AuthorizeView>(0);
            __builder.AddAttribute(1, "ChildContent", (Microsoft.AspNetCore.Components.RenderFragment<Microsoft.AspNetCore.Components.Authorization.AuthenticationState>)((context) => (__builder2) => {
#nullable restore
#line 6 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
     if(!isTraining && !isDoneTraining) {

#line default
#line hidden
#nullable disable
                __builder2.OpenElement(2, "div");
                __builder2.AddMarkupContent(3, "<h1>Please select a location to train</h1>\n        ");
                __builder2.OpenComponent<GoogleMapsComponents.GoogleMap>(4);
                __builder2.AddAttribute(5, "Id", "map1");
                __builder2.AddAttribute(6, "Options", Microsoft.AspNetCore.Components.CompilerServices.RuntimeHelpers.TypeCheck<GoogleMapsComponents.Maps.MapOptions>(
#nullable restore
#line 9 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                                    mapOptions

#line default
#line hidden
#nullable disable
                ));
                __builder2.AddAttribute(7, "OnAfterInit", Microsoft.AspNetCore.Components.CompilerServices.RuntimeHelpers.TypeCheck<Microsoft.AspNetCore.Components.EventCallback>(Microsoft.AspNetCore.Components.EventCallback.Factory.Create(this, 
#nullable restore
#line 9 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                                                               async () => await OnAfterInitAsync()

#line default
#line hidden
#nullable disable
                )));
                __builder2.AddComponentReferenceCapture(8, (__value) => {
#nullable restore
#line 9 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                          map1 = (GoogleMapsComponents.GoogleMap)__value;

#line default
#line hidden
#nullable disable
                }
                );
                __builder2.CloseComponent();
                __builder2.AddMarkupContent(9, "\n        <br>\n        ");
                __builder2.OpenComponent<Microsoft.AspNetCore.Components.Forms.EditForm>(10);
                __builder2.AddAttribute(11, "Model", Microsoft.AspNetCore.Components.CompilerServices.RuntimeHelpers.TypeCheck<System.Object>(
#nullable restore
#line 12 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                               request

#line default
#line hidden
#nullable disable
                ));
                __builder2.AddAttribute(12, "ChildContent", (Microsoft.AspNetCore.Components.RenderFragment<Microsoft.AspNetCore.Components.Forms.EditContext>)((another_one) => (__builder3) => {
                    __Blazor.WeatherApp.Server.Pages.Map.TypeInference.CreateInputRadioGroup_0(__builder3, 13, 14, 
#nullable restore
#line 13 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                          request.ModelType

#line default
#line hidden
#nullable disable
                    , 15, Microsoft.AspNetCore.Components.EventCallback.Factory.Create(this, Microsoft.AspNetCore.Components.CompilerServices.RuntimeHelpers.CreateInferredEventCallback(this, __value => request.ModelType = __value, request.ModelType)), 16, () => request.ModelType, 17, (__builder4) => {
                        __Blazor.WeatherApp.Server.Pages.Map.TypeInference.CreateInputRadio_1(__builder4, 18, 19, 
#nullable restore
#line 14 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                   1

#line default
#line hidden
#nullable disable
                        );
                        __builder4.AddMarkupContent(20, " LSTM (Fast)<br>\n                ");
                        __Blazor.WeatherApp.Server.Pages.Map.TypeInference.CreateInputRadio_2(__builder4, 21, 22, 
#nullable restore
#line 15 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                   2

#line default
#line hidden
#nullable disable
                        );
                        __builder4.AddMarkupContent(23, " Stacked LSTM (Balanced)<br>\n                ");
                        __Blazor.WeatherApp.Server.Pages.Map.TypeInference.CreateInputRadio_3(__builder4, 24, 25, 
#nullable restore
#line 16 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                   3

#line default
#line hidden
#nullable disable
                        );
                        __builder4.AddMarkupContent(26, " Biredictional (Thorough)<br>");
                    }
                    );
                }
                ));
                __builder2.CloseComponent();
                __builder2.AddMarkupContent(27, "\n        <br>\n        ");
                __builder2.OpenElement(28, "button");
                __builder2.AddAttribute(29, "class", "btn-danger");
                __builder2.AddAttribute(30, "onclick", Microsoft.AspNetCore.Components.EventCallback.Factory.Create<Microsoft.AspNetCore.Components.Web.MouseEventArgs>(this, 
#nullable restore
#line 20 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                             Train

#line default
#line hidden
#nullable disable
                ));
                __builder2.AddContent(31, " Train Location");
                __builder2.CloseElement();
                __builder2.CloseElement();
#nullable restore
#line 22 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
    }
    else if(!isTraining && isDoneTraining && !hasFailed)
    {

#line default
#line hidden
#nullable disable
                __builder2.AddMarkupContent(32, "<h3>Results</h3>\r\n        ");
                __builder2.OpenElement(33, "h4");
                __builder2.AddContent(34, "Weather Forcast for tomorrow is: ");
                __builder2.AddContent(35, 
#nullable restore
#line 26 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                              tmwWeather

#line default
#line hidden
#nullable disable
                );
                __builder2.AddMarkupContent(36, " °C");
                __builder2.CloseElement();
                __builder2.AddMarkupContent(37, "\r\n        ");
                __builder2.OpenElement(38, "button");
                __builder2.AddAttribute(39, "class", "btn-danger");
                __builder2.AddAttribute(40, "onclick", Microsoft.AspNetCore.Components.EventCallback.Factory.Create<Microsoft.AspNetCore.Components.Web.MouseEventArgs>(this, 
#nullable restore
#line 27 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                             Refresh

#line default
#line hidden
#nullable disable
                ));
                __builder2.AddContent(41, "Try different region");
                __builder2.CloseElement();
                __builder2.AddMarkupContent(42, "\r\n        <br>\r\n        ");
                __builder2.OpenElement(43, "img");
                __builder2.AddAttribute(44, "src", 
#nullable restore
#line 29 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                   imageResult

#line default
#line hidden
#nullable disable
                );
                __builder2.AddAttribute(45, "width", "80%");
                __builder2.AddAttribute(46, "height", "50%");
                __builder2.CloseElement();
#nullable restore
#line 30 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
    }
    else if(!isTraining && isDoneTraining && hasFailed)
    {

#line default
#line hidden
#nullable disable
                __builder2.AddMarkupContent(47, "<h3>Could not find any data for this location, please try again</h3>\r\n        ");
                __builder2.OpenElement(48, "button");
                __builder2.AddAttribute(49, "class", "btn-primary");
                __builder2.AddAttribute(50, "onclick", Microsoft.AspNetCore.Components.EventCallback.Factory.Create<Microsoft.AspNetCore.Components.Web.MouseEventArgs>(this, 
#nullable restore
#line 34 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
                                              Refresh

#line default
#line hidden
#nullable disable
                ));
                __builder2.AddContent(51, "Try different region");
                __builder2.CloseElement();
#nullable restore
#line 35 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
    }
    else if(isTraining) {

#line default
#line hidden
#nullable disable
                __builder2.AddMarkupContent(52, "<h3>please wait.... training model</h3>");
#nullable restore
#line 38 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
    }

#line default
#line hidden
#nullable disable
            }
            ));
            __builder.CloseComponent();
        }
        #pragma warning restore 1998
#nullable restore
#line 42 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp.Server\Pages\Map.razor"
        
    private GoogleMap map1 = default!;
    private MapOptions mapOptions = default!;
    private WeatherApp.Logic.WeatherPythonApi api = new Logic.WeatherPythonApi();
    private WeatherApp.Logic.WeatherRequest request = new Logic.WeatherRequest();
    private bool isTraining = false;
    private bool isDoneTraining = false;
    private bool hasFailed = false;

    private Stack<Marker> markers = new Stack<Marker>();
    private LatLngLiteral latLng = null;
    private string labelText = "";
    private string imageResult = "";
    private float tmwWeather = 0;

    protected void Refresh()
    {
        isDoneTraining = false;
        StateHasChanged();
    }

    protected async Task Train()
    {
        imageResult = "";
        hasFailed = false;
        isDoneTraining = false;
        isTraining = true;
        StateHasChanged();

        if (latLng == null)
        {
            return;
        }
        request.Lat = latLng.Lat;
        request.Lon = latLng.Lng;

        var results = await api.SendTrainLocation(request);
        
        if(results.Reason == "")
        {
            imageResult = results.ResultImage;
            tmwWeather = (float) Math.Round(results.WeatherPredictedTomorrow, 1);
            Console.WriteLine(imageResult);
        }
        else
        {
            hasFailed = true;
        }

        isDoneTraining = true;
        isTraining = false;

        StateHasChanged();

    }

    protected override void OnInitialized()
    {
        mapOptions = new MapOptions()
        {
            Zoom = 5,
            Center = new LatLngLiteral()
            {
                Lat = 52,
                Lng = 0
            },
            MapTypeId = MapTypeId.Roadmap
        };
    }

    private async Task OnAfterInitAsync()
    {
        await map1.InteropObject.AddListener<MouseEvent>("click", async (e) => await OnClick(e));
    }


    private async Task OnClick(MouseEvent e)
    {
        await RemoveMarker();
        await AddMarker(e.LatLng);

        latLng = e.LatLng;
    }


    private async Task AddMarker(LatLngLiteral latling)
    {
        var marker = await Marker.CreateAsync(map1.JsRuntime,
            new MarkerOptions()
            {
                Position = latling,
                Map = map1.InteropObject,
                Draggable = false,
                Icon = new Icon()
                {
                    Url = "https://www.pngall.com/wp-content/uploads/2017/05/Map-Marker-PNG-File.png",
                    ScaledSize = new Size { Height = 35, Width = 35}
                }
            });

        var icon = await marker.GetIcon();

        Console.WriteLine($"Get icon result type is : {icon.Value.GetType()}");

        icon.Switch(
            s => Console.WriteLine(s),
            i => Console.WriteLine(i.Url),
            _ => { });

        markers.Push(marker);
        labelText = await marker.GetLabelText();

        await marker.AddListener<MouseEvent>("click", async e =>
        {
            string markerLabelText = await marker.GetLabelText();
            StateHasChanged();
            await e.Stop();
        });
        await marker.AddListener<MouseEvent>("dragend", async e => await OnMakerDragEnd(marker, e));
    }

    private async Task OnMakerDragEnd(Marker M, MouseEvent e)
    {
        string markerLabelText = await M.GetLabelText();
        StateHasChanged();
        await e.Stop();
    }

    private async Task RemoveMarker()
    {
        if (!markers.Any())
        {
            return;
        }

        var lastMarker = markers.Pop();
        await lastMarker.SetMap(null);
        labelText = markers.Any() ? await markers.Peek().GetLabelText() : "";
    }

#line default
#line hidden
#nullable disable
    }
}
namespace __Blazor.WeatherApp.Server.Pages.Map
{
    #line hidden
    internal static class TypeInference
    {
        public static void CreateInputRadioGroup_0<TValue>(global::Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder, int seq, int __seq0, TValue __arg0, int __seq1, global::Microsoft.AspNetCore.Components.EventCallback<TValue> __arg1, int __seq2, global::System.Linq.Expressions.Expression<global::System.Func<TValue>> __arg2, int __seq3, global::Microsoft.AspNetCore.Components.RenderFragment __arg3)
        {
        __builder.OpenComponent<global::Microsoft.AspNetCore.Components.Forms.InputRadioGroup<TValue>>(seq);
        __builder.AddAttribute(__seq0, "Value", __arg0);
        __builder.AddAttribute(__seq1, "ValueChanged", __arg1);
        __builder.AddAttribute(__seq2, "ValueExpression", __arg2);
        __builder.AddAttribute(__seq3, "ChildContent", __arg3);
        __builder.CloseComponent();
        }
        public static void CreateInputRadio_1<TValue>(global::Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder, int seq, int __seq0, TValue __arg0)
        {
        __builder.OpenComponent<global::Microsoft.AspNetCore.Components.Forms.InputRadio<TValue>>(seq);
        __builder.AddAttribute(__seq0, "Value", __arg0);
        __builder.CloseComponent();
        }
        public static void CreateInputRadio_2<TValue>(global::Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder, int seq, int __seq0, TValue __arg0)
        {
        __builder.OpenComponent<global::Microsoft.AspNetCore.Components.Forms.InputRadio<TValue>>(seq);
        __builder.AddAttribute(__seq0, "Value", __arg0);
        __builder.CloseComponent();
        }
        public static void CreateInputRadio_3<TValue>(global::Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder, int seq, int __seq0, TValue __arg0)
        {
        __builder.OpenComponent<global::Microsoft.AspNetCore.Components.Forms.InputRadio<TValue>>(seq);
        __builder.AddAttribute(__seq0, "Value", __arg0);
        __builder.CloseComponent();
        }
    }
}
#pragma warning restore 1591
