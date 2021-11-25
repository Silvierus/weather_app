// <auto-generated/>
#pragma warning disable 1591
#pragma warning disable 0414
#pragma warning disable 0649
#pragma warning disable 0169

namespace WeatherApp.Pages
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Components;
#nullable restore
#line 1 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using System.Net.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using System.Net.Http.Json;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using Microsoft.AspNetCore.Components.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using Microsoft.AspNetCore.Components.Forms;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using Microsoft.AspNetCore.Components.Routing;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using Microsoft.AspNetCore.Components.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 7 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using Microsoft.AspNetCore.Components.Web.Virtualization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 8 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using Microsoft.AspNetCore.Components.WebAssembly.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 9 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#nullable disable
#nullable restore
#line 10 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using WeatherApp;

#line default
#line hidden
#nullable disable
#nullable restore
#line 11 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\_Imports.razor"
using WeatherApp.Shared;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\Pages\Map.razor"
using GoogleMapsComponents;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\Pages\Map.razor"
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
        }
        #pragma warning restore 1998
#nullable restore
#line 14 "C:\Users\silvi\source\repos\WeatherApp\WeatherApp\Pages\Map.razor"
        private GoogleMap map1 = default!;
    private MapOptions mapOptions = default!;
    private WeatherApp.Logic.WeatherPythonApi api = new Logic.WeatherPythonApi();
    private List<String> _events = new List<String>();

    private bool DisablePoiInfoWindow { get; set; } = false;

    private Stack<Marker> markers = new Stack<Marker>();
    private string labelText = "";

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
        //todo: when click connect to python endpoint and do stuff
        await api.SendTrainLocation(new Logic.WeatherRequest
        {
            Lat = 50,
            Lon = 0
        });

        // "lat,lon"

        _events.Insert(0, $"Click {e.LatLng}.");
        _events = _events.Take(100).ToList();

        StateHasChanged();

        if (DisablePoiInfoWindow)
        {
            await e.Stop();
        }
    }


    private async Task AddMarker()
    {
        var marker = await Marker.CreateAsync(map1.JsRuntime,
            new MarkerOptions()
            {
                Position = await map1.InteropObject.GetCenter(),
                Map = map1.InteropObject,
                Label = new MarkerLabel { Text = $"Test {markers.Count()}", FontWeight = "bold" },
                Draggable = true,
                Icon = new Icon()
                {
                    Url = "https://developers.google.com/maps/documentation/javascript/examples/full/images/beachflag.png"
                }
                //Icon = "https://developers.google.com/maps/documentation/javascript/examples/full/images/beachflag.png"
            });

        //await marker.SetMap(map1);

        //var map = await marker.GetMap();

        var icon = await marker.GetIcon();

        Console.WriteLine($"Get icon result type is : {icon.Value.GetType()}");

        icon.Switch(
            s => Console.WriteLine(s),
            i => Console.WriteLine(i.Url),
            _ => { });

        //if (map == map1.InteropObject)
        //{
        //    Console.WriteLine("Yess");
        //}
        //else
        //{
        //    Console.WriteLine("Nooo");
        //}

        markers.Push(marker);
        labelText = await marker.GetLabelText();

        await marker.AddListener<MouseEvent>("click", async e =>
        {
            string markerLabelText = await marker.GetLabelText();
            _events.Add("click on " + markerLabelText);
            StateHasChanged();
            await e.Stop();
        });
        await marker.AddListener<MouseEvent>("dragend", async e => await OnMakerDragEnd(marker, e));
    }

    private async Task OnMakerDragEnd(Marker M, MouseEvent e)
    {
        string markerLabelText = await M.GetLabelText();
        _events.Insert(0, $"OnMakerDragEnd ({markerLabelText}): ({e.LatLng}).");
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
#pragma warning restore 1591
